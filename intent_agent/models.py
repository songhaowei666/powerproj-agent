"""Pydantic 模型定义。"""

from typing import List
from pydantic import BaseModel, Field


class TaskPlan(BaseModel):
    """单个任务规划。"""

    task_id: str = Field(..., description="任务ID，如 task_1, task_2")
    business: str = Field(
        ..., description="业务类型：统计业务/规划业务/投资业务"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="置信度 0-1"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的任务ID列表"
    )
    description: str = Field(..., description="任务描述")


class IntentResult(BaseModel):
    """意图识别结果。"""

    tasks: List[TaskPlan] = Field(..., description="任务规划列表")
    reasoning: str = Field(..., description="推理过程说明")
