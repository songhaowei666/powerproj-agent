"""Planning Agent A2A Server - 基于 DefaultRequestHandler + AgentExecutor。

使用 a2a_base.get_a2a_app 创建 Starlette app，额外挂载 /files/{file_id} 文件下载路由。
"""

import base64
from pathlib import Path
from typing import Any, Dict, List

from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2
from a2a.helpers import new_text_message, get_message_text

from providers.llm_provider import get_llm
from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.graph import build_planning_graph
from planning_agent.models import PlanningState


# ---------- Agent Card (protobuf) ----------


def _build_agent_card() -> a2a_pb2.AgentCard:
    """构造 protobuf AgentCard。"""
    card = a2a_pb2.AgentCard()
    card.name = "planning-agent"
    card.description = "规划业务 Agent，负责电力项目信息查询、节点文件上传/下载/删除"
    card.version = "1.1.0"

    # capabilities
    card.capabilities.streaming = False
    card.capabilities.push_notifications = False

    # default modes
    card.default_input_modes.append("text")
    card.default_output_modes.append("text")
    card.default_output_modes.append("file")

    # skills
    skill1 = card.skills.add()
    skill1.id = "project-query"
    skill1.name = "项目信息查询"
    skill1.description = "根据自然语言查询电力项目基本信息及聚合统计"
    skill1.tags.append("planning")
    skill1.tags.append("project")
    skill1.tags.append("query")
    skill1.examples.append("查一下北京西500千伏项目的信息")
    skill1.examples.append("所有项目变电容量的总和是多少")
    skill1.examples.append("线路长度超过100公里的220kv项目有哪些")

    skill2 = card.skills.add()
    skill2.id = "file-management"
    skill2.name = "节点文件管理"
    skill2.description = "按节点编码（001可研设计/002可研评审/003可研批复）上传、下载、删除文件"
    skill2.tags.append("planning")
    skill2.tags.append("file")
    skill2.tags.append("upload")
    skill2.tags.append("download")
    skill2.examples.append("上传可研设计文件到北京西项目")
    skill2.examples.append("下载北京西项目的可研评审文件")
    skill2.examples.append("删除北京西项目可研设计节点的报告.pdf")

    # supported interfaces
    iface = card.supported_interfaces.add()
    iface.protocol_binding = "JSONRPC"
    iface.url = "http://localhost:8001"

    return card


AGENT_CARD = _build_agent_card()


# ---------- Agent Executor ----------


class PlanningAgentExecutor(AgentExecutor):
    """Planning Agent Executor。

    基于 LangGraph 实现，支持 interrupt / resume 多轮对话。
    """

    def __init__(self, llm=None, db=None, fm=None):
        self._llm = llm or get_llm()
        self._db = db or ProjectDatabase()
        self._fm = fm or FileManager()
        self._graph = build_planning_graph(self._llm, self._db, self._fm)

    @staticmethod
    def _extract_files_from_message(message) -> List[Dict[str, Any]]:
        """从 protobuf Message 中提取上传的文件。"""
        files = []
        if message is None:
            return files
        for part in message.parts:
            content_type = part.WhichOneof("content")
            if content_type == "raw" and part.raw:
                try:
                    files.append(
                        {
                            "name": part.filename or "unnamed",
                            "content": bytes(part.raw),
                            "mime_type": part.media_type,
                        }
                    )
                except Exception:
                    pass
        return files

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """取消任务。"""
        updater = TaskUpdater(
            event_queue, context.task_id, context.context_id
        )
        cancel_msg = new_text_message(
            text="任务已取消",
            context_id=context.context_id,
            task_id=context.task_id,
        )
        await updater.cancel(cancel_msg)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """执行 A2A 任务。

        逻辑：
        1. 提取 message 中的文本和文件
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
        pending_files = self._extract_files_from_message(message)

        # 将 bytes 转为 base64 字符串，避免 checkpointer 序列化失败
        for f in pending_files:
            if isinstance(f.get("content"), bytes):
                f["content"] = base64.b64encode(f["content"]).decode()

        # SDK 要求：必须先发送一个 Task 事件，然后才能发送 TaskStatusUpdateEvent
        initial_task = a2a_pb2.Task()
        initial_task.id = task_id
        initial_task.context_id = context_id
        initial_task.status.state = a2a_pb2.TaskState.TASK_STATE_SUBMITTED
        await event_queue.enqueue_event(initial_task)

        if not current_text and not pending_files:
            updater = TaskUpdater(event_queue, task_id, context_id)
            error_msg = new_text_message(
                text="无法从消息中提取有效内容",
                context_id=context_id,
                task_id=task_id,
            )
            await updater.failed(error_msg)
            return

        config = {"configurable": {"thread_id": task_id}}

        # 检查 graph 当前状态
        graph_state = await self._graph.aget_state(config)

        if graph_state and graph_state.next:
            # 图处于中断状态 -> 恢复执行
            from langgraph.types import Command

            result = await self._graph.ainvoke(Command(resume=current_text), config)
        else:
            # 新请求
            initial_state = PlanningState(
                query=current_text,
                pending_files=pending_files,
            )
            result = await self._graph.ainvoke(initial_state, config)

        updater = TaskUpdater(event_queue, task_id, context_id)

        # 再次检查是否仍为中断状态
        graph_state = await self._graph.aget_state(config)
        if graph_state and graph_state.next:
            try:
                interrupt_info = graph_state.tasks[0].interrupts[0].value
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
        result_state = PlanningState.model_validate(result)

        if result_state.status == "failed":
            error_text = result_state.result_text or "未知错误"
            error_message = new_text_message(
                text=error_text,
                context_id=context_id,
                task_id=task_id,
            )
            await updater.failed(error_message)
            return

        # 发送 artifacts（如下载文件时的链接列表）
        for artifact_dict in result_state.artifacts:
            parts = []
            if artifact_dict.get("type") == "text":
                part = a2a_pb2.Part()
                part.text = artifact_dict.get("text", "")
                parts.append(part)
            elif artifact_dict.get("type") == "file":
                file_info = artifact_dict.get("file", {})
                part = a2a_pb2.Part()
                part.url = file_info.get("uri", "")
                part.filename = file_info.get("name", "")
                parts.append(part)
            if parts:
                await updater.add_artifact(parts=parts)

        # 组装完成消息
        final_text = result_state.result_text or "操作已完成"
        final_message = new_text_message(
            text=final_text,
            context_id=context_id,
            task_id=task_id,
        )
        await updater.complete(final_message)


# ---------- 文件下载路由 ----------


db_instance = ProjectDatabase()
fm_instance = FileManager()


async def download_file(request):
    """文件下载 handler。"""
    file_id = request.path_params["file_id"]
    file_info = db_instance.get_file_by_id(file_id)
    if not file_info:
        return JSONResponse(
            status_code=404, content={"error": f"File not found: {file_id}"}
        )
    file_path = Path(file_info["file_path"])
    if not file_path.exists():
        return JSONResponse(
            status_code=404, content={"error": f"File not found on disk: {file_id}"}
        )
    return FileResponse(
        path=str(file_path),
        filename=file_info.get("file_name", file_path.name),
    )


EXTRA_ROUTES = [
    Route("/files/{file_id}", download_file, methods=["GET"]),
]
