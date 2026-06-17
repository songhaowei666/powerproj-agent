"""Pydantic 模型定义。"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SubTask(BaseModel):
    """子任务定义。"""

    id: str = Field(..., description="子任务编号，如 task_1, task_2")
    name: str = Field(..., description="子任务名称")
    description: str = Field(..., description="子任务详细描述")
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的子任务 ID 列表"
    )
    expected_output: str = Field(..., description="预期输出描述")
    required_agent: str = Field(
        ..., min_length=1, description="目标业务 Agent 名称，必须匹配某个 AgentCard 的 name"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="置信度 0-1"
    )

    @field_validator("required_agent")
    @classmethod
    def _validate_required_agent(cls, value: str) -> str:
        """去除首尾空白，并拒绝空 Agent 名称。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("required_agent 不能为空")
        return normalized


class TaskPlan(BaseModel):
    """任务规划结果。"""

    task_goal: str = Field(..., description="原始任务目标概述")
    subtasks: List[SubTask] = Field(..., description="子任务列表")
    execution_order: List[str] = Field(
        ..., description="按执行顺序排列的子任务 ID"
    )


class IntentResult(BaseModel):
    """意图识别结果。"""

    is_business_query: bool = Field(
        default=True,
        description="是否为电网业务相关查询；问候、闲聊、与业务无关的内容为 false",
    )
    task_goal: str = Field(..., description="原始任务目标概述")
    subtasks: List[SubTask] = Field(..., description="子任务列表")
    execution_order: List[str] = Field(
        ..., description="按执行顺序排列的子任务 ID"
    )
    reasoning: str = Field(..., description="推理过程说明")
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="需要用户补充信息时，面向用户的澄清问句；非业务或信息已充分时为 null",
    )
