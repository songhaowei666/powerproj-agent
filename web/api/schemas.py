"""Web API 请求与响应模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str = Field(..., min_length=1, description="用户消息内容")
    task_id: Optional[str] = Field(default=None, description="续传任务 ID")
    context_id: Optional[str] = Field(default=None, description="续传 context ID")
    base_url: Optional[str] = Field(default=None, description="主控 Agent 地址")


class ConnectivityResponse(BaseModel):
    """连通性检查响应。"""

    online: bool
    base_url: str


class ConfirmationOptionSchema(BaseModel):
    """确认按钮选项。"""

    id: str
    label: str
    replyText: str


class ConfirmationSchema(BaseModel):
    """结构化确认交互。"""

    action: str
    title: str = ""
    options: List[ConfirmationOptionSchema]
    confirm_type: str = "confirmation"
    body: Optional[Dict[str, Any]] = None


class ChatResponseSchema(BaseModel):
    """聊天完成响应。"""

    task_id: str
    context_id: str
    state: str
    text: str
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    invocation_traces: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    confirmation: Optional[ConfirmationSchema] = None
