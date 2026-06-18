"""Task Manager 单元测试。"""

from intent_agent.models import IntentResult, SubTask

from main_agent.task_manager import (
    TaskManager,
    format_plan_approve_reply,
    parse_plan_approve_selection,
    parse_plan_confirm_action,
)
from main_agent.task_models import ManagedTaskPlan


def _build_intent_result() -> IntentResult:
    return IntentResult(
        task_goal="查询项目并下载文件",
        subtasks=[
            SubTask(
                id="task_1",
                name="查询项目基本信息",
                description="查询北京西500千伏项目基本信息",
                dependencies=[],
                expected_output="项目基本信息",
                required_agent="planning-agent",
            ),
            SubTask(
                id="task_2",
                name="下载可研设计文件",
                description="下载可研设计节点文件",
                dependencies=["task_1"],
                expected_output="文件链接",
                required_agent="planning-agent",
            ),
        ],
        execution_order=["task_1", "task_2"],
        reasoning="测试",
    )


class TestTaskManager:
    """TaskManager 状态流转测试。"""

    def test_create_plan_from_intent(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        assert plan.plan_status == "draft"
        assert plan.revision == 1
        assert len(plan.tasks) == 2
        assert all(task.status == "pending" for task in plan.tasks)
        assert plan.events[-1].event_type == "plan_created"

    def test_approve_and_start_execution(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        approved = TaskManager.approve_plan(plan)
        assert approved.plan_status == "approved"
        assert approved.approved_at is not None

        executing = TaskManager.start_execution(approved)
        assert executing.plan_status == "executing"

    def test_task_progress_updates(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        plan = TaskManager.start_execution(TaskManager.approve_plan(plan))
        plan = TaskManager.mark_task_in_progress(plan, "task_1")
        plan = TaskManager.mark_task_completed(plan, "task_1")
        plan = TaskManager.mark_task_in_progress(plan, "task_2")

        payload = TaskManager.build_progress_payload(plan)
        assert payload["completed_count"] == 1
        assert payload["total_count"] == 2
        assert payload["current_task_id"] == "task_2"
        assert payload["tasks"][0]["status"] == "completed"
        assert payload["tasks"][1]["status"] == "in_progress"

    def test_cancel_plan(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        plan = TaskManager.mark_task_in_progress(plan, "task_1")
        cancelled = TaskManager.cancel_plan(plan, "用户取消")
        assert cancelled.plan_status == "cancelled"
        assert cancelled.tasks[0].status == "skipped"
        assert cancelled.tasks[1].status == "skipped"

    def test_revise_plan_preserves_completed(self):
        old_plan = TaskManager.create_plan_from_intent(_build_intent_result())
        old_plan = TaskManager.mark_task_completed(old_plan, "task_1")

        new_intent = IntentResult(
            task_goal="只要项目信息",
            subtasks=[
                SubTask(
                    id="task_1",
                    name="查询项目基本信息",
                    description="查询项目",
                    dependencies=[],
                    expected_output="项目信息",
                    required_agent="planning-agent",
                )
            ],
            execution_order=["task_1"],
            reasoning="修改后",
        )
        revised = TaskManager.revise_plan(old_plan, new_intent)
        assert revised.revision == 2
        assert revised.tasks[0].status == "completed"
        assert revised.events[-1].event_type == "plan_revised"

    def test_build_plan_confirm_body(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        body = TaskManager.build_plan_confirm_body(plan)
        assert body["goal"] == "查询项目并下载文件"
        assert len(body["tasks"]) == 2
        assert body["tasks"][0]["required_agent"] == "planning-agent"

    def test_parse_plan_confirm_action(self):
        assert parse_plan_confirm_action("确认执行") == "approve"
        assert parse_plan_confirm_action("确认执行:task_1,task_2") == "approve"
        assert parse_plan_confirm_action("取消") == "cancel"
        assert parse_plan_confirm_action("修改计划：不要下载") == "modify"

    def test_parse_plan_approve_selection(self):
        assert parse_plan_approve_selection("确认执行") is None
        assert parse_plan_approve_selection("确认执行:task_1,task_2") == [
            "task_1",
            "task_2",
        ]
        assert parse_plan_approve_selection("确认执行:") == []

    def test_format_plan_approve_reply(self):
        assert format_plan_approve_reply(["task_1", "task_2"], ["task_1", "task_2"]) == "确认执行"
        assert (
            format_plan_approve_reply(["task_1"], ["task_1", "task_2"])
            == "确认执行:task_1"
        )

    def test_apply_task_selection(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        updated = TaskManager.apply_task_selection(plan, ["task_1"])
        assert updated.tasks[0].status == "pending"
        assert updated.tasks[1].status == "skipped"

    def test_build_cancel_message(self):
        plan = TaskManager.create_plan_from_intent(_build_intent_result())
        plan = TaskManager.mark_task_completed(plan, "task_1")
        plan = TaskManager.cancel_plan(plan, "用户取消")
        message = TaskManager.build_cancel_message(plan)
        assert "已完成 1/2" in message
        assert "task_1" in message
        assert "已跳过" in message
