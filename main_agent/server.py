"""
主控 Agent (Main Agent) A2A Server
用户请求入口，负责任务调度与 orchestration。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from a2a_base import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Task,
    TaskStatus,
    Message,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
)
from providers.llm_provider import get_llm
from main_agent.graph import build_main_graph
from main_agent.models import MainState

# ---------- Agent 配置 ----------

AGENT_CARD = AgentCard(
    name="main-agent",
    description="主控 Agent，用户请求的统一入口，负责任务调度与 orchestration",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="task-orchestration",
            name="任务调度",
            description="识别用户意图，按依赖关系分阶段并行调度业务 Agent 执行任务",
            tags=["orchestration", "main"],
            examples=[
                "帮我统计今年的投资收益并做明年规划",
                "分析上月数据并制定下月计划",
            ],
        ),
    ],
)

# ---------- 全局实例 ----------

_llm = get_llm()
_graph = build_main_graph(_llm)

# 会话存储：task_id -> 用户消息历史（用于 query 拼接）
_sessions: Dict[str, list] = {}


# ---------- 工具函数 ----------


def _extract_text_from_task(task: Task) -> str:
    """从 Task 的 message 中提取文本内容。"""
    message = task.history[-1] if task.history else None
    if not message:
        return ""
    parts = message.parts
    if not parts:
        return ""
    # 拼接所有 text part
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
    return "\n".join(texts)


def _build_input_required_response(task: Task, question: str) -> Task:
    """构造需要用户补充信息的响应。"""
    task.status = TaskStatus(
        state="input-required",
        message=Message(
            role="agent",
            parts=[{"type": "text", "text": question}],
        ),
    )
    task.artifacts = [{"type": "text", "text": question}]
    return task


def _build_completed_response(task: Task, artifacts: list) -> Task:
    """构造执行完成的响应。"""
    task.status = TaskStatus(
        state="completed",
        message=Message(
            role="agent",
            parts=[{"type": "text", "text": "任务执行完成"}],
        ),
    )
    task.artifacts = artifacts
    return task


def _build_failed_response(task: Task, error_message: str) -> Task:
    """构造执行失败的响应。"""
    task.status = TaskStatus(
        state="failed",
        message=Message(
            role="agent",
            parts=[{"type": "text", "text": error_message}],
        ),
    )
    task.artifacts = [{"type": "text", "text": error_message}]
    return task


# ---------- 异步 Handler ----------


async def _async_handle_task(task: Task) -> Task:
    """异步处理 A2A 任务。

    逻辑：
    1. 提取当前 message 文本
    2. 使用 task.id 作为 LangGraph thread_id
    3. 检查 graph 状态：
       - 若处于中断：用 Command(resume=text) 恢复
       - 若未开始：初始化 MainState 启动
    4. 再次检查状态，组装 A2A 响应
    """
    task_id = task.id
    current_text = _extract_text_from_task(task)

    if not current_text:
        return _build_failed_response(task, "无法从消息中提取文本内容")

    # 维护会话消息历史
    if task_id not in _sessions:
        _sessions[task_id] = []
    _sessions[task_id].append(current_text)

    config = {"configurable": {"thread_id": task_id}}

    # 检查 graph 当前状态
    state = await _graph.aget_state(config)

    if state and state.next:
        # 图处于中断状态，用户发来了补充信息 -> 恢复执行
        from langgraph.types import Command

        result = await _graph.ainvoke(Command(resume=current_text), config)
    else:
        # 新请求
        full_query = "\n".join(_sessions[task_id])
        initial_state = MainState(
            query=full_query,
            session_id=task.sessionId or task_id,
        )
        result = await _graph.ainvoke(initial_state, config)

    # 再次检查是否仍为中断状态
    state = await _graph.aget_state(config)
    if state and state.next:
        # 提取 interrupt 信息
        try:
            interrupt_info = state.tasks[0].interrupts[0].value
            question = interrupt_info.get("question", "请补充信息")
        except (IndexError, AttributeError):
            question = "请补充更多信息"
        return _build_input_required_response(task, question)

    # 图已结束，根据结果组装响应
    result_dict = result if isinstance(result, dict) else result.model_dump()
    status = result_dict.get("status", "completed")

    if status == "failed":
        error_msg = result_dict.get("error_message", "未知错误")
        return _build_failed_response(task, error_msg)

    artifacts = result_dict.get("final_artifacts", [])
    return _build_completed_response(task, artifacts)


# ---------- JSON-RPC 路由 ----------


app = FastAPI(title="main-agent", version="1.0.0")

# 内存任务存储
_tasks_store: Dict[str, Task] = {}


@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump(exclude_none=True)


@app.post("/")
async def jsonrpc_endpoint(req: Request):
    body = await req.json()
    rpc_req = JSONRPCRequest(**body)
    response = await _handle_rpc_async(rpc_req)
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def _handle_rpc_async(rpc_req: JSONRPCRequest) -> JSONRPCResponse:
    method = rpc_req.method
    params = rpc_req.params or {}
    req_id = rpc_req.id

    if method == "tasks/send":
        return await _tasks_send_async(params, req_id)
    elif method == "tasks/get":
        return _tasks_get(params, req_id)
    elif method == "tasks/cancel":
        return _tasks_cancel(params, req_id)
    else:
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32601, message=f"Method not found: {method}"),
        )


async def _tasks_send_async(params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
    import uuid

    task_id = params.get("id") or str(uuid.uuid4())
    session_id = params.get("sessionId")
    message_data = params.get("message", {})
    metadata = params.get("metadata")

    message = Message(
        role=message_data.get("role", "user"),
        parts=message_data.get("parts", []),
    )

    # 如果 task_id 已存在，追加 message 到 history
    if task_id in _tasks_store:
        existing = _tasks_store[task_id]
        existing.history.append(message)
        task = existing
    else:
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(state="submitted"),
            history=[message],
            metadata=metadata,
        )

    task = await _async_handle_task(task)
    _tasks_store[task_id] = task
    return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))


def _tasks_get(params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
    task_id = params.get("id")
    task = _tasks_store.get(task_id)
    if not task:
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32001, message=f"Task not found: {task_id}"),
        )
    return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))


def _tasks_cancel(params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
    task_id = params.get("id")
    task = _tasks_store.get(task_id)
    if not task:
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=-32001, message=f"Task not found: {task_id}"),
        )
    task.status.state = "canceled"
    return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
