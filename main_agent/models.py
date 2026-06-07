"""Pydantic 模型定义 - Main Agent 状态与任务输出。"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from intent_agent.models import IntentResult


class TaskOutput(BaseModel):
    """单个业务任务的执行结果。"""

    task_id: str = Field(..., description="任务ID")
    business: str = Field(..., description="业务类型")
    status: str = Field(..., description="执行状态：success / failed")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="业务Agent返回的artifacts")
    error: Optional[str] = Field(default=None, description="失败时的错误信息")


class MainState(BaseModel):
    """Main Agent LangGraph 状态。"""

    query: str = Field(default="", description="当前完整查询（含补充信息）")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    intent_result: Optional[IntentResult] = Field(default=None, description="意图识别结果")
    phases: List[List[str]] = Field(
        default_factory=list, description="拓扑分层结果，每个元素是同层task_id列表"
    )
    current_phase_idx: int = Field(default=0, description="当前执行阶段索引")
    task_outputs: Dict[str, TaskOutput] = Field(
        default_factory=dict, description="各任务执行结果"
    )
    failed_task_id: Optional[str] = Field(default=None, description="失败的任务ID")
    error_message: Optional[str] = Field(default=None, description="失败时的错误信息")
    final_artifacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="最终汇总的artifacts"
    )
    status: str = Field(
        default="pending", description="整体状态：pending / executing / completed / failed"
    )
    summary: Optional[str] = Field(
        default=None, description="LLM 生成的自然语言总结"
    )
