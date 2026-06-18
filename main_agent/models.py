"""Pydantic 模型定义 - Main Agent 状态与任务输出。"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from intent_agent.models import IntentResult
from main_agent.task_models import ManagedTaskPlan


class InvocationTraceEntry(BaseModel):
    """单次 Agent 调用的轨迹记录。"""

    step: int = Field(..., description="调用序号，从 1 开始")
    agent_type: str = Field(..., description="Agent 类型：intent / business")
    agent_name: str = Field(..., description="Agent 名称")
    capability: Optional[str] = Field(default=None, description="目标业务 Agent 名称")
    phase: Optional[int] = Field(default=None, description="业务 Agent 所在执行阶段")
    task_id: Optional[str] = Field(default=None, description="子任务 ID")
    input: Dict[str, Any] = Field(default_factory=dict, description="调用入参")
    output: Dict[str, Any] = Field(default_factory=dict, description="调用出参")
    status: str = Field(default="success", description="调用状态：success / failed")


class TaskOutput(BaseModel):
    """单个子任务的执行结果。"""

    task_id: str = Field(..., description="任务ID")
    required_agent: str = Field(..., description="目标业务 Agent 名称，对应 AgentCard.name")
    status: str = Field(..., description="执行状态：success / failed")
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="业务Agent返回的artifacts"
    )
    error: Optional[str] = Field(default=None, description="失败时的错误信息")


class MainState(BaseModel):
    """Main Agent LangGraph 状态。"""

    query: str = Field(default="", description="当前完整查询（含补充信息）")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    intent_result: Optional[IntentResult] = Field(
        default=None, description="意图识别结果"
    )
    phases: List[List[str]] = Field(
        default_factory=list, description="拓扑分层结果，每个元素是同层subtask_id列表"
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
        default="pending",
        description="整体状态：pending / executing / completed / failed / cancelled",
    )
    summary: Optional[str] = Field(
        default=None, description="LLM 生成的自然语言总结"
    )
    invocation_traces: List[Dict[str, Any]] = Field(
        default_factory=list, description="意图识别与业务 Agent 的调用轨迹"
    )
    task_plan: Optional[ManagedTaskPlan] = Field(
        default=None, description="Task Manager 托管的计划与进度"
    )
    cancel_requested: bool = Field(default=False, description="协作式取消标志")
    plan_revision_base: Optional[int] = Field(
        default=None, description="用户修改计划时的版本基数"
    )
    replan_from_modify: bool = Field(
        default=False, description="是否因修改计划回到意图识别"
    )
