"""Main Agent Executor - 实现 a2a.server.agent_execution.AgentExecutor。

参考 planning_agent/executor.py 的实现方式，使用 A2A SDK 的 TaskUpdater 与
EventQueue 与客户端交互，内部通过 LangGraph 完成意图识别与任务调度。
"""

from typing import Dict, List, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2
from a2a.helpers import new_text_message, get_message_text
from langchain_core.language_models import BaseChatModel
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from main_agent.agent_network import AgentNetwork
from main_agent.graph import build_main_graph
from main_agent.models import MainState


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
    def _build_query_from_history(task: a2a_pb2.Task) -> str:
        """从 A2A Task 的 history 中提取所有用户消息文本并拼接。

        A2A SDK 的 InMemoryTaskStore 会自动维护 task history，因此无需在 Executor
        中自行维护会话状态。
        """
        texts: List[str] = []
        for message in task.history:
            if getattr(message, "role", None) != "user":
                continue
            text = get_message_text(message)
            if text:
                texts.append(text)
        return "\n".join(texts)

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

        # SDK 要求：必须先发送一个 Task 事件，然后才能发送 TaskStatusUpdateEvent
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

        full_query = self._build_query_from_history(task)
        config = {"configurable": {"thread_id": task_id}}
        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            # 检查 graph 当前状态
            state = await self._graph.aget_state(config)

            if state and state.next:
                # 图处于中断状态，用户发来了补充信息 -> 恢复执行
                result = await self._graph.ainvoke(
                    Command(resume=current_text), config
                )
            else:
                # 新请求
                initial_state = MainState(
                    query=full_query,
                    session_id=task.session_id or task_id,
                )
                result = await self._graph.ainvoke(initial_state, config)
        except GraphInterrupt:
            # 执行过程中触发 interrupt -> 返回 input-required
            pass

        # 再次检查是否仍为中断状态
        state = await self._graph.aget_state(config)
        if state and state.next:
            try:
                interrupt_info = state.tasks[0].interrupts[0].value
                question = interrupt_info.get("question", "请补充信息")
            except (IndexError, AttributeError):
                question = "请补充更多信息"

            status_message = new_text_message(
                text=question,
                context_id=context_id,
                task_id=task_id,
            )
            await updater.update_status(
                a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED, status_message
            )
            return

        # 图已结束，根据结果组装响应
        result_dict = result if isinstance(result, dict) else result.model_dump()
        status = result_dict.get("status", "completed")

        if status == "failed":
            error_msg = result_dict.get("error_message", "未知错误")
            error_message = new_text_message(
                text=error_msg,
                context_id=context_id,
                task_id=task_id,
            )
            await updater.failed(error_message)
            return

        # 发送 artifacts
        final_artifacts = result_dict.get("final_artifacts", [])
        for artifact in final_artifacts:
            parts = self._artifact_to_parts(artifact)
            if parts:
                await updater.add_artifact(parts=parts)

        final_text = result_dict.get("summary") or "任务执行完成"
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
                if art.get("type") == "text":
                    lines.append(art.get("text", ""))
                elif art.get("type") == "file" or "url" in art:
                    lines.append(f"文件：{art.get('url', '')}")
            part = a2a_pb2.Part()
            part.text = "\n".join(lines)
            parts.append(part)

        return parts
