"""Main Agent Executor - 实现 a2a.server.agent_execution.AgentExecutor。

参考 planning_agent/executor.py 的实现方式，使用 A2A SDK 的 TaskUpdater 与
EventQueue 与客户端交互，内部通过 LangGraph 完成意图识别与任务调度。
"""

import asyncio
import json
from typing import Awaitable, Callable, Dict, List, Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2
from a2a.helpers import new_text_message, get_message_text
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from a2a_message_parser import build_agent_message_from_parts
from main_agent.agent_network import AgentNetwork
from main_agent.graph import build_main_graph
from main_agent.models import MainState
from main_agent.executor import extract_artifact_text
from main_agent.streaming import format_summary_chunk_message, format_trace_step_message


TracePublisher = Callable[[Dict[str, Any]], Awaitable[None]]
SummaryChunkPublisher = Callable[[str], Awaitable[None]]


class _StreamPublisher:
    """将 LangGraph 内部事件转为 A2A WORKING 状态流式推送。"""

    def __init__(
        self,
        updater: TaskUpdater,
        context_id: str,
        task_id: str,
    ) -> None:
        self._updater = updater
        self._context_id = context_id
        self._task_id = task_id
        self._lock = asyncio.Lock()
        self._working_sent = False

    async def _ensure_working(self) -> None:
        if self._working_sent:
            return
        working_message = new_text_message(
            text="正在处理您的请求...",
            context_id=self._context_id,
            task_id=self._task_id,
        )
        await self._updater.update_status(
            a2a_pb2.TaskState.TASK_STATE_WORKING,
            working_message,
        )
        self._working_sent = True

    async def publish_trace(self, trace_dict: Dict[str, Any]) -> None:
        """推送单条调用轨迹。"""
        async with self._lock:
            await self._ensure_working()
            trace_message = new_text_message(
                text=format_trace_step_message(trace_dict),
                context_id=self._context_id,
                task_id=self._task_id,
            )
            await self._updater.update_status(
                a2a_pb2.TaskState.TASK_STATE_WORKING,
                trace_message,
            )

    async def publish_summary_chunk(self, chunk: str) -> None:
        """推送总结文本分块。"""
        if not chunk:
            return
        async with self._lock:
            await self._ensure_working()
            chunk_message = new_text_message(
                text=format_summary_chunk_message(chunk),
                context_id=self._context_id,
                task_id=self._task_id,
            )
            await self._updater.update_status(
                a2a_pb2.TaskState.TASK_STATE_WORKING,
                chunk_message,
            )


class MainAgentExecutor(AgentExecutor):
    """主控 Agent Executor。

    接收用户 A2A 请求，复用 A2A SDK 的 Task 历史管理，调用 LangGraph 完成意图识别与多 Agent 调度。
    """

    def __init__(self, llm: BaseChatModel, agent_network: AgentNetwork):
        self._llm = llm
        self._agent_network = agent_network
        self._graph = build_main_graph(llm, agent_network)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """取消任务。"""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        cancel_msg = new_text_message(
            text="任务已取消",
            context_id=context.context_id,
            task_id=context.task_id,
        )
        await updater.cancel(cancel_msg)

    @staticmethod
    def _build_query_from_history(
        task: Optional[a2a_pb2.Task],
        fallback_text: str = "",
    ) -> str:
        """从 A2A Task 的 history 中提取所有用户消息文本并拼接。

        A2A SDK 的 InMemoryTaskStore 会自动维护 task history，因此无需在 Executor
        中自行维护会话状态。首次对话时 ``context.current_task`` 可能为 None，
        此时回退到当前消息文本。
        """
        if task is None:
            return fallback_text

        texts: List[str] = []
        for message in task.history:
            if message.role != a2a_pb2.ROLE_USER:
                continue
            text = get_message_text(message)
            if text:
                texts.append(text)
        return "\n".join(texts) if texts else fallback_text

    @staticmethod
    def _state_values_to_dict(state_values: Any) -> Dict[str, Any]:
        """将 LangGraph checkpoint 中的状态值转为 dict。"""
        if state_values is None:
            return {}
        if isinstance(state_values, dict):
            return state_values
        if hasattr(state_values, "model_dump"):
            return state_values.model_dump()
        return {}

    @staticmethod
    def _build_graph_config(
        task_id: str,
        stream_publisher: _StreamPublisher,
    ) -> Dict[str, Any]:
        """构建 LangGraph 运行配置，注入流式推送回调。"""
        return {
            "configurable": {
                "thread_id": task_id,
                "publish_trace": stream_publisher.publish_trace,
                "publish_summary_chunk": stream_publisher.publish_summary_chunk,
            }
        }

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """执行 A2A 任务。

        逻辑：
        1. 提取 message 文本
        2. 使用 task_id 作为 LangGraph thread_id
        3. 检查 graph 状态，决定恢复或新建
        4. 若 graph 中断，发送 input-required 状态
        5. 若完成，发送 completed / failed 状态
        """
        task = context.current_task
        message = context.message
        task_id = context.task_id
        context_id = context.context_id

        current_text = get_message_text(message) if message else ""

        # 仅首次创建 Task 时发送 SUBMITTED 事件，避免重复 task 告警
        if task is None:
            initial_task = a2a_pb2.Task()
            initial_task.id = task_id
            initial_task.context_id = context_id
            initial_task.status.state = a2a_pb2.TaskState.TASK_STATE_SUBMITTED
            await event_queue.enqueue_event(initial_task)

        if not current_text:
            updater = TaskUpdater(event_queue, task_id, context_id)
            error_msg = new_text_message(
                text="无法从消息中提取文本内容",
                context_id=context_id,
                task_id=task_id,
            )
            await updater.failed(error_msg)
            return

        full_query = self._build_query_from_history(task, current_text)
        updater = TaskUpdater(event_queue, task_id, context_id)
        stream_publisher = _StreamPublisher(updater, context_id, task_id)
        config = self._build_graph_config(task_id, stream_publisher)
        invoke_result: Any = None

        try:
            # 检查 graph 当前状态
            state = await self._graph.aget_state(config)

            if state and state.next:
                # 图处于中断状态，用户发来了补充信息 -> 恢复执行
                invoke_result = await self._graph.ainvoke(
                    Command(resume=current_text), config
                )
            else:
                # 新请求
                session_id = task.session_id if task and task.session_id else task_id
                initial_state = MainState(
                    query=full_query,
                    session_id=session_id,
                )
                invoke_result = await self._graph.ainvoke(initial_state, config)
        except GraphInterrupt:
            # 执行过程中触发 interrupt -> 返回 input-required
            pass

        # 再次检查是否仍为中断状态
        state = await self._graph.aget_state(config)
        if state and state.next:
            try:
                interrupt_info = state.tasks[0].interrupts[0].value
                question = interrupt_info.get("question", "请补充信息")
                parts = interrupt_info.get("parts") or []
            except (IndexError, AttributeError):
                question = "请补充更多信息"
                parts = []

            if parts:
                status_message = build_agent_message_from_parts(
                    parts, context_id, task_id
                )
            else:
                status_message = new_text_message(
                    text=question,
                    context_id=context_id,
                    task_id=task_id,
                )
            await updater.update_status(
                a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED, status_message
            )
            return

        # 图已结束，优先从 checkpoint 读取完整状态（resume 后 ainvoke 返回值可能不完整）
        result_dict = self._state_values_to_dict(state.values if state else None)
        if not result_dict:
            result_dict = self._state_values_to_dict(invoke_result)
        status = result_dict.get("status", "completed")

        if status == "failed":
            error_msg = result_dict.get("error_message", "未知错误")
            # 失败路径保留完整轨迹 artifact，供非流式客户端兜底
            failed_traces = result_dict.get("invocation_traces", [])
            if failed_traces:
                trace_parts = self._artifact_to_parts(
                    {
                        "type": "invocation_trace",
                        "traces": failed_traces,
                    }
                )
                if trace_parts:
                    await updater.add_artifact(parts=trace_parts)
            error_message = new_text_message(
                text=error_msg,
                context_id=context_id,
                task_id=task_id,
            )
            await updater.failed(error_message)
            return

        # 发送 artifacts（成功路径不再推送 invocation_trace，轨迹已在 WORKING 阶段流式发送）
        final_artifacts = result_dict.get("final_artifacts", [])
        for artifact in final_artifacts:
            if artifact.get("type") == "invocation_trace":
                continue
            parts = self._artifact_to_parts(artifact)
            if parts:
                await updater.add_artifact(parts=parts)

        final_text = (result_dict.get("summary") or "").strip()
        if not final_text:
            artifact_texts: List[str] = []
            for artifact in result_dict.get("final_artifacts", []):
                if artifact.get("type") == "text" and artifact.get("text"):
                    artifact_texts.append(artifact["text"])
            final_text = (
                "\n\n".join(artifact_texts)
                if artifact_texts
                else "未产生任何执行结果。"
            )
        final_message = new_text_message(
            text=final_text,
            context_id=context_id,
            task_id=task_id,
        )
        await updater.complete(final_message)

    @staticmethod
    def _artifact_to_parts(artifact: Dict[str, Any]) -> List[a2a_pb2.Part]:
        """将内部 artifact dict 转换为 protobuf Part 列表。"""
        artifact_type = artifact.get("type")
        parts: List[a2a_pb2.Part] = []

        if artifact_type == "text":
            part = a2a_pb2.Part()
            part.text = artifact.get("text", "")
            parts.append(part)
        elif artifact_type == "file":
            part = a2a_pb2.Part()
            part.url = artifact.get("url", "")
            part.filename = artifact.get("name", artifact.get("filename", "文件"))
            parts.append(part)
        elif artifact_type == "task_result":
            # 将 task_result 以文本形式展示
            lines = [f"【任务 {artifact.get('task_id', '')}】"]
            for art in artifact.get("artifacts", []):
                text = extract_artifact_text(art)
                if text:
                    lines.append(text)
                    continue
                for part in art.get("parts", []):
                    url = part.get("url", "")
                    if url:
                        lines.append(f"文件：{url}")
            part = a2a_pb2.Part()
            part.text = "\n".join(lines)
            parts.append(part)
        elif artifact_type == "invocation_trace":
            part = a2a_pb2.Part()
            part.text = (
                "__INVOCATION_TRACE__\n"
                + json.dumps(artifact.get("traces", []), ensure_ascii=False, indent=2)
            )
            parts.append(part)

        return parts
