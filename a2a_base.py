"""
A2A 协议基础服务器实现
基于 Google A2A Protocol (JSON-RPC over HTTP)
提供通用的 AgentCard、Task 管理和 JSON-RPC 端点
"""

import uuid
import time
from typing import Dict, Callable, Any, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ---------- Pydantic 模型定义 (兼容 A2A 协议子集) ----------

class TextPart(BaseModel):
    type: str = "text"
    text: str


class Message(BaseModel):
    role: str  # "user" | "agent"
    parts: list[Dict[str, Any]]


class TaskStatus(BaseModel):
    state: str  # submitted, working, completed, canceled, failed...
    message: Optional[Message] = None


class Task(BaseModel):
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: list[Dict[str, Any]] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class AgentSkill(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentCard(BaseModel):
    name: str
    description: Optional[str] = None
    url: str
    provider: Optional[Dict[str, str]] = None
    version: str = "1.0.0"
    documentationUrl: Optional[str] = None
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    authentication: Optional[Dict[str, Any]] = None
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text"])


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[Any] = None


# ---------- A2A Server 核心类 ----------

class A2AServer:
    """
    通用 A2A Agent 服务器
    只需传入 agent_card 和 task_handler 即可快速启动一个符合 A2A 协议的测试服务
    """

    def __init__(
        self,
        agent_card: AgentCard,
        task_handler: Callable[[Task], Task],
    ):
        self.agent_card = agent_card
        self.task_handler = task_handler
        self.tasks: Dict[str, Task] = {}
        self.app = FastAPI(title=agent_card.name, version=agent_card.version)
        self._register_routes()

    def _register_routes(self):
        app = self.app

        @app.get("/.well-known/agent.json")
        async def get_agent_card():
            return self.agent_card.model_dump(exclude_none=True)

        @app.post("/")
        async def jsonrpc_endpoint(req: Request):
            body = await req.json()
            rpc_req = JSONRPCRequest(**body)
            response = self._handle_rpc(rpc_req)
            return JSONResponse(content=response.model_dump(exclude_none=True))

    def _handle_rpc(self, rpc_req: JSONRPCRequest) -> JSONRPCResponse:
        method = rpc_req.method
        params = rpc_req.params or {}
        req_id = rpc_req.id

        if method == "tasks/send":
            return self._tasks_send(params, req_id)
        elif method == "tasks/get":
            return self._tasks_get(params, req_id)
        elif method == "tasks/cancel":
            return self._tasks_cancel(params, req_id)
        else:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32601, message=f"Method not found: {method}"),
            )

    def _tasks_send(self, params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
        task_id = params.get("id") or str(uuid.uuid4())
        session_id = params.get("sessionId")
        message_data = params.get("message", {})
        metadata = params.get("metadata")

        # 构建 Message
        message = Message(
            role=message_data.get("role", "user"),
            parts=message_data.get("parts", []),
        )

        # 创建 Task（初始状态 submitted）
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(state="submitted"),
            history=[message],
            metadata=metadata,
        )

        # 交给业务 handler 处理
        task = self.task_handler(task)

        # 保存任务
        self.tasks[task.id] = task

        return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))

    def _tasks_get(self, params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
        task_id = params.get("id")
        task = self.tasks.get(task_id)
        if not task:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32001, message=f"Task not found: {task_id}"),
            )
        return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))

    def _tasks_cancel(self, params: Dict[str, Any], req_id: Any) -> JSONRPCResponse:
        task_id = params.get("id")
        task = self.tasks.get(task_id)
        if not task:
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=-32001, message=f"Task not found: {task_id}"),
            )
        task.status.state = "canceled"
        return JSONRPCResponse(id=req_id, result=task.model_dump(exclude_none=True))


def create_a2a_app(agent_card: AgentCard, task_handler: Callable[[Task], Task]) -> FastAPI:
    """快捷函数：传入 AgentCard 和 handler，直接拿到 FastAPI app"""
    server = A2AServer(agent_card=agent_card, task_handler=task_handler)
    return server.app
