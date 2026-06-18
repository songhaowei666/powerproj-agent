"""任务计划与进度管理器。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from intent_agent.models import IntentResult, SubTask

from main_agent.task_models import ManagedTask, ManagedTaskPlan, ProgressEvent

PLAN_CONFIRM_QUESTION = "我已为您制定以下执行计划，请确认是否开始执行："

PLAN_CONFIRM_OPTIONS: List[Dict[str, str]] = [
    {"id": "approve", "label": "开始执行", "replyText": "确认执行"},
    {"id": "modify", "label": "修改计划", "replyText": "修改计划："},
    {"id": "cancel", "label": "取消", "replyText": "取消"},
]


def _utc_now_iso() -> str:
    """返回 UTC ISO8601 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _clone_plan(plan: ManagedTaskPlan) -> ManagedTaskPlan:
    """深拷贝计划对象，避免原地修改副作用。"""
    return deepcopy(plan)


def _append_event(
    plan: ManagedTaskPlan,
    event_type: str,
    *,
    task_id: Optional[str] = None,
    message: str = "",
) -> None:
    """向计划追加进度事件。"""
    plan.events.append(
        ProgressEvent(
            timestamp=_utc_now_iso(),
            event_type=event_type,
            task_id=task_id,
            message=message,
            revision=plan.revision,
        )
    )


def _managed_task_from_subtask(subtask: SubTask) -> ManagedTask:
    """将 Intent SubTask 转为 ManagedTask。"""
    return ManagedTask(
        id=subtask.id,
        name=subtask.name,
        description=subtask.description,
        dependencies=list(subtask.dependencies),
        expected_output=subtask.expected_output,
        required_agent=subtask.required_agent,
        status="pending",
    )


def parse_plan_confirm_action(user_reply: str) -> str:
    """解析计划确认 interrupt 的用户回复。

    Returns:
        approve | modify | cancel | unknown
    """
    text = (user_reply or "").strip()
    if not text:
        return "unknown"
    if text in {"确认执行", "开始执行"} or text.startswith("确认执行:"):
        return "approve"
    if text == "取消":
        return "cancel"
    if text.startswith("修改计划"):
        return "modify"
    return "unknown"


def parse_plan_approve_selection(user_reply: str) -> Optional[List[str]]:
    """解析计划确认时用户勾选的子任务 ID 列表。

    Returns:
        None 表示全选；非空列表表示部分选中；空列表表示未选任何任务。
    """
    text = (user_reply or "").strip()
    if text in {"确认执行", "开始执行"}:
        return None
    prefix = "确认执行:"
    if not text.startswith(prefix):
        return None
    raw_ids = text[len(prefix) :].strip()
    if not raw_ids:
        return []
    return [item.strip() for item in raw_ids.split(",") if item.strip()]


def format_plan_approve_reply(
    selected_task_ids: List[str],
    all_task_ids: List[str],
) -> str:
    """构建带勾选结果的确认执行回复文本。"""
    if set(selected_task_ids) == set(all_task_ids):
        return "确认执行"
    return f"确认执行:{','.join(selected_task_ids)}"


def extract_plan_modify_text(user_reply: str) -> str:
    """从修改计划回复中提取用户补充说明。"""
    text = (user_reply or "").strip()
    prefix = "修改计划："
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    if text.startswith("修改计划"):
        return text[len("修改计划") :].strip(" ：:")
    return text


class TaskManager:
    """任务计划与进度管理器。"""

    @staticmethod
    def create_plan_from_intent(
        intent_result: IntentResult,
        *,
        revision: int = 1,
    ) -> ManagedTaskPlan:
        """将 IntentResult 转为 draft 态 ManagedTaskPlan。"""
        plan = ManagedTaskPlan(
            revision=revision,
            goal=intent_result.task_goal,
            tasks=[_managed_task_from_subtask(item) for item in intent_result.subtasks],
            plan_status="draft",
        )
        _append_event(plan, "plan_created", message="已生成执行计划")
        return plan

    @staticmethod
    def approve_plan(plan: ManagedTaskPlan) -> ManagedTaskPlan:
        """用户确认执行：draft -> approved。"""
        updated = _clone_plan(plan)
        updated.plan_status = "approved"
        updated.approved_at = _utc_now_iso()
        _append_event(updated, "plan_approved", message="用户已确认执行计划")
        return updated

    @staticmethod
    def apply_task_selection(
        plan: ManagedTaskPlan,
        selected_task_ids: List[str],
    ) -> ManagedTaskPlan:
        """将未勾选的 pending 子任务标记为 skipped。"""
        selected_set = set(selected_task_ids)
        updated = _clone_plan(plan)
        for task in updated.tasks:
            if task.id not in selected_set and task.status == "pending":
                task.status = "skipped"
        return updated

    @staticmethod
    def cancel_plan(plan: ManagedTaskPlan, reason: str = "") -> ManagedTaskPlan:
        """取消计划并将 pending / in_progress 任务标为 skipped。"""
        updated = _clone_plan(plan)
        updated.plan_status = "cancelled"
        updated.cancel_reason = reason or "用户取消"
        for task in updated.tasks:
            if task.status in {"pending", "in_progress"}:
                task.status = "skipped"
        _append_event(
            updated,
            "plan_cancelled",
            message=updated.cancel_reason,
        )
        return updated

    @staticmethod
    def start_execution(plan: ManagedTaskPlan) -> ManagedTaskPlan:
        """进入执行：approved -> executing。"""
        updated = _clone_plan(plan)
        updated.plan_status = "executing"
        return updated

    @staticmethod
    def mark_plan_completed(plan: ManagedTaskPlan) -> ManagedTaskPlan:
        """标记计划全部完成。"""
        updated = _clone_plan(plan)
        updated.plan_status = "completed"
        return updated

    @staticmethod
    def mark_plan_failed(plan: ManagedTaskPlan) -> ManagedTaskPlan:
        """标记计划执行失败。"""
        updated = _clone_plan(plan)
        updated.plan_status = "failed"
        return updated

    @staticmethod
    def mark_task_in_progress(plan: ManagedTaskPlan, task_id: str) -> ManagedTaskPlan:
        """子任务开始执行。"""
        updated = _clone_plan(plan)
        task = updated.get_task(task_id)
        if task is None:
            return updated
        task.status = "in_progress"
        task.started_at = _utc_now_iso()
        _append_event(updated, "task_started", task_id=task_id, message=task.name)
        return updated

    @staticmethod
    def mark_task_completed(plan: ManagedTaskPlan, task_id: str) -> ManagedTaskPlan:
        """子任务执行成功。"""
        updated = _clone_plan(plan)
        task = updated.get_task(task_id)
        if task is None:
            return updated
        task.status = "completed"
        task.completed_at = _utc_now_iso()
        task.error = None
        _append_event(updated, "task_completed", task_id=task_id, message=task.name)
        return updated

    @staticmethod
    def mark_task_failed(
        plan: ManagedTaskPlan,
        task_id: str,
        error: str,
    ) -> ManagedTaskPlan:
        """子任务执行失败。"""
        updated = _clone_plan(plan)
        task = updated.get_task(task_id)
        if task is None:
            return updated
        task.status = "failed"
        task.completed_at = _utc_now_iso()
        task.error = error
        _append_event(
            updated,
            "task_failed",
            task_id=task_id,
            message=error,
        )
        return updated

    @staticmethod
    def mark_remaining_skipped(plan: ManagedTaskPlan) -> ManagedTaskPlan:
        """取消时将剩余 pending / in_progress 任务标为 skipped。"""
        updated = _clone_plan(plan)
        for task in updated.tasks:
            if task.status in {"pending", "in_progress"}:
                task.status = "skipped"
        return updated

    @staticmethod
    def revise_plan(
        plan: ManagedTaskPlan,
        new_intent_result: IntentResult,
        *,
        preserve_completed: bool = True,
    ) -> ManagedTaskPlan:
        """改计划：revision +1，合并新 IntentResult。"""
        completed_map: Dict[str, ManagedTask] = {}
        if preserve_completed:
            for task in plan.tasks:
                if task.status == "completed":
                    completed_map[task.id] = task

        updated = TaskManager.create_plan_from_intent(
            new_intent_result,
            revision=plan.revision + 1,
        )
        merged_tasks: List[ManagedTask] = []
        for task in updated.tasks:
            previous = completed_map.get(task.id)
            if previous is not None:
                merged_tasks.append(deepcopy(previous))
            else:
                merged_tasks.append(task)
        updated.tasks = merged_tasks
        _append_event(updated, "plan_revised", message="用户修改了执行计划")
        return updated

    @staticmethod
    def build_progress_payload(plan: ManagedTaskPlan) -> Dict[str, Any]:
        """构建 __TASK_PROGRESS__ 流式推送 JSON。"""
        completed_count = sum(1 for task in plan.tasks if task.status == "completed")
        current_task_id: Optional[str] = None
        for task in plan.tasks:
            if task.status == "in_progress":
                current_task_id = task.id
                break
        if current_task_id is None:
            for task in plan.tasks:
                if task.status == "pending":
                    current_task_id = task.id
                    break

        return {
            "revision": plan.revision,
            "plan_status": plan.plan_status,
            "goal": plan.goal,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "required_agent": task.required_agent,
                }
                for task in plan.tasks
            ],
            "completed_count": completed_count,
            "total_count": len(plan.tasks),
            "current_task_id": current_task_id,
        }

    @staticmethod
    def build_plan_confirm_body(plan: ManagedTaskPlan) -> Dict[str, Any]:
        """构建 plan_confirm interrupt 的 body 字段。"""
        return plan.to_plan_summary_dict()

    @staticmethod
    def build_plan_confirm_text(plan: ManagedTaskPlan) -> str:
        """构建计划确认消息的文本兜底内容。"""
        return TaskManager.build_plan_confirm_text_from_summary(
            plan.to_plan_summary_dict()
        )

    @staticmethod
    def build_plan_confirm_text_from_summary(plan_summary: Dict[str, Any]) -> str:
        """根据计划 summary dict 构建文本兜底内容。"""
        lines = [PLAN_CONFIRM_QUESTION, ""]
        for index, task in enumerate(plan_summary.get("tasks", []), start=1):
            agent_name = task.get("required_agent", "")
            task_name = task.get("name", "")
            lines.append(f"{index}. [{agent_name}] {task_name}")
        return "\n".join(lines)

    @staticmethod
    def build_cancel_message(plan: Optional[ManagedTaskPlan]) -> str:
        """构建取消结果说明文本。"""
        if plan is None:
            return "任务已取消。"

        total = len(plan.tasks)
        completed_tasks = [task for task in plan.tasks if task.status == "completed"]
        skipped_tasks = [task for task in plan.tasks if task.status == "skipped"]
        failed_tasks = [task for task in plan.tasks if task.status == "failed"]

        lines = [f"任务已取消。已完成 {len(completed_tasks)}/{total} 个子任务："]
        for task in plan.tasks:
            if task.status == "completed":
                lines.append(f"- {task.id} {task.name}：成功")
            elif task.status == "failed":
                lines.append(f"- {task.id} {task.name}：失败")
            elif task.status == "skipped":
                lines.append(f"- {task.id} {task.name}：已跳过")
        if not completed_tasks and not skipped_tasks and not failed_tasks:
            lines.append("- 尚未开始执行任何子任务")
        return "\n".join(lines)
