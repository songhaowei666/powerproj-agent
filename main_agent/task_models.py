"""Task Manager 数据模型定义。"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ManagedTaskStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
    "skipped",
]

ManagedPlanStatus = Literal[
    "draft",
    "approved",
    "executing",
    "completed",
    "cancelled",
    "failed",
]


class ManagedTask(BaseModel):
    """托管子任务，由 Intent SubTask 初始化并附加运行时状态。"""

    id: str
    name: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    expected_output: str
    required_agent: str
    confidence: float = 1.0
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ProgressEvent(BaseModel):
    """进度时间线事件，供 UI 渲染。"""

    timestamp: str
    event_type: str
    task_id: Optional[str] = None
    message: str = ""
    revision: int = 1


class ManagedTaskPlan(BaseModel):
    """Task Manager 托管的完整计划。"""

    revision: int = 1
    goal: str
    tasks: List[ManagedTask]
    plan_status: str = "draft"
    approved_at: Optional[str] = None
    cancel_reason: Optional[str] = None
    events: List[ProgressEvent] = Field(default_factory=list)

    def get_task(self, task_id: str) -> Optional[ManagedTask]:
        """按 ID 查找子任务。"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def to_plan_summary_dict(self) -> Dict[str, Any]:
        """导出计划确认/进度展示用的精简结构。"""
        return {
            "revision": self.revision,
            "goal": self.goal,
            "plan_status": self.plan_status,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "required_agent": task.required_agent,
                    "dependencies": list(task.dependencies),
                    "status": task.status,
                }
                for task in self.tasks
            ],
        }
