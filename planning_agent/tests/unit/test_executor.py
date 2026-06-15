"""executor.py 单元测试。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_message_parser import ParsedInput, parse_message_parts
from planning_agent.executor import PlanningAgentExecutor
from planning_agent.models import PlanningState


@pytest.fixture
def mock_llm():
    """提供 mock LLM。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="SELECT COUNT(*) AS cnt"))
    return llm


@pytest.fixture
def executor(mock_llm):
    """提供 mock LLM 构造的 executor。"""
    return PlanningAgentExecutor(llm=mock_llm)


class TestBuildPendingFiles:
    """测试 raw 文件解析与转换。"""

    def test_build_pending_files_from_parsed(self, executor):
        """通过共享解析器提取 raw 文件。"""
        message = MagicMock()
        part = MagicMock()
        part.WhichOneof.return_value = "raw"
        part.raw = b"file content"
        part.filename = "test.pdf"
        part.media_type = "application/pdf"
        message.parts = [part]

        parsed = parse_message_parts(message)
        files = executor._build_pending_files(parsed.raw_files)
        assert len(files) == 1
        assert files[0]["name"] == "test.pdf"
        assert files[0]["mime_type"] == "application/pdf"
        assert isinstance(files[0]["content"], str)

    def test_build_pending_files_empty(self, executor):
        assert executor._build_pending_files([]) == []


class TestCancel:
    """测试 cancel。"""

    @pytest.mark.asyncio
    async def test_cancel_sends_cancelled_message(self, executor):
        """cancel 发送任务已取消消息。"""
        context = MagicMock()
        context.task_id = "task-1"
        context.context_id = "ctx-1"

        event_queue = AsyncMock()

        with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater_cls.return_value = mock_updater
            mock_updater.cancel = AsyncMock()

            await executor.cancel(context, event_queue)

            mock_updater_cls.assert_called_once_with(
                event_queue, "task-1", "ctx-1"
            )
            mock_updater.cancel.assert_awaited_once()


class TestExecute:
    """测试 execute。"""

    @pytest.mark.asyncio
    async def test_execute_empty_message_fails(self, executor):
        """空文本和文件时返回 failed。"""
        context = MagicMock()
        context.task_id = "task-empty"
        context.context_id = "ctx-empty"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater_cls.return_value = mock_updater
            mock_updater.failed = AsyncMock()

            # 模拟 get_message_text 返回空字符串且无文件
            with patch(
                "planning_agent.executor.parse_message_parts",
                return_value=ParsedInput(task_query="", raw_files=[]),
            ):
                await executor.execute(context, event_queue)

            mock_updater.failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_aggregate_query_completes(self, executor):
        """聚合查询完成后返回 completed。"""
        context = MagicMock()
        context.task_id = "task-agg"
        context.context_id = "ctx-agg"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        with patch("planning_agent.executor.get_message_text", return_value="所有项目数量"):
            with patch.object(
                executor._graph,
                "aget_state",
                new=AsyncMock(return_value=None),
            ):
                with patch.object(
                    executor._graph,
                    "ainvoke",
                    new=AsyncMock(
                        return_value=PlanningState(
                            intent="query_project",
                            status="completed",
                            result_text="共 10 个项目",
                            artifacts=[],
                            project_confirmed=True,
                        ).model_dump()
                    ),
                ):
                    with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
                        mock_updater = MagicMock()
                        mock_updater_cls.return_value = mock_updater
                        mock_updater.complete = AsyncMock()
                        mock_updater.add_artifact = AsyncMock()

                        await executor.execute(context, event_queue)

                        mock_updater.complete.assert_awaited_once()
                        call_args = mock_updater.complete.await_args
                        assert "共 10 个项目" in str(call_args)

    @pytest.mark.asyncio
    async def test_execute_returns_input_required_on_interrupt(self, executor):
        """graph 中断时返回 input-required。"""
        context = MagicMock()
        context.task_id = "task-interrupt"
        context.context_id = "ctx-interrupt"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        graph_state = MagicMock()
        graph_state.next = ("confirm_project",)
        task_obj = MagicMock()
        interrupt = MagicMock()
        interrupt.value = {"question": "请问是这个项目吗？"}
        task_obj.interrupts = [interrupt]
        graph_state.tasks = [task_obj]

        with patch("planning_agent.executor.get_message_text", return_value="查一下北京西项目"):
            with patch.object(
                executor._graph,
                "aget_state",
                new=AsyncMock(side_effect=[None, graph_state]),
            ):
                with patch.object(
                    executor._graph,
                    "ainvoke",
                    new=AsyncMock(return_value={}),
                ):
                    with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
                        mock_updater = MagicMock()
                        mock_updater_cls.return_value = mock_updater
                        mock_updater.update_status = AsyncMock()

                        await executor.execute(context, event_queue)

                        mock_updater.update_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_graph_interrupt_exception(self, executor):
        """ainvoke 抛出 GraphInterrupt 时仍返回 input-required。"""
        from langgraph.errors import GraphInterrupt

        context = MagicMock()
        context.task_id = "task-graph-interrupt"
        context.context_id = "ctx-graph-interrupt"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        graph_state = MagicMock()
        graph_state.next = ("confirm_project",)
        task_obj = MagicMock()
        interrupt = MagicMock()
        interrupt.value = {"question": "请确认"}
        task_obj.interrupts = [interrupt]
        graph_state.tasks = [task_obj]

        with patch("planning_agent.executor.get_message_text", return_value="查一下"):
            with patch.object(
                executor._graph,
                "aget_state",
                new=AsyncMock(side_effect=[None, graph_state]),
            ):
                with patch.object(
                    executor._graph,
                    "ainvoke",
                    new=AsyncMock(side_effect=GraphInterrupt()),
                ):
                    with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
                        mock_updater = MagicMock()
                        mock_updater_cls.return_value = mock_updater
                        mock_updater.update_status = AsyncMock()

                        await executor.execute(context, event_queue)

                        mock_updater.update_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_interrupt_info_missing_fallback(self, executor):
        """interrupt 信息缺失时使用默认提示。"""
        context = MagicMock()
        context.task_id = "task-missing-interrupt"
        context.context_id = "ctx-missing-interrupt"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        graph_state = MagicMock()
        graph_state.next = ("confirm_project",)
        task_obj = MagicMock()
        task_obj.interrupts = []  # 空列表触发 IndexError
        graph_state.tasks = [task_obj]

        with patch("planning_agent.executor.get_message_text", return_value="查一下"):
            with patch.object(
                executor._graph,
                "aget_state",
                new=AsyncMock(side_effect=[None, graph_state]),
            ):
                with patch.object(
                    executor._graph,
                    "ainvoke",
                    new=AsyncMock(return_value={}),
                ):
                    with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
                        mock_updater = MagicMock()
                        mock_updater_cls.return_value = mock_updater
                        mock_updater.update_status = AsyncMock()

                        await executor.execute(context, event_queue)

                        mock_updater.update_status.assert_awaited_once()
                        call_args = mock_updater.update_status.await_args
                        assert "请补充更多信息" in str(call_args)

    @pytest.mark.asyncio
    async def test_execute_failed_state(self, executor):
        """graph 返回 failed 状态时发送 failed 消息。"""
        context = MagicMock()
        context.task_id = "task-fail"
        context.context_id = "ctx-fail"
        context.message = MagicMock()
        context.current_task = MagicMock()

        event_queue = AsyncMock()

        with patch("planning_agent.executor.get_message_text", return_value="随便说"):
            with patch.object(
                executor._graph,
                "aget_state",
                new=AsyncMock(return_value=None),
            ):
                with patch.object(
                    executor._graph,
                    "ainvoke",
                    new=AsyncMock(
                        return_value=PlanningState(
                            intent="unknown",
                            status="failed",
                            result_text="无法理解您的意图",
                        ).model_dump()
                    ),
                ):
                    with patch("planning_agent.executor.TaskUpdater") as mock_updater_cls:
                        mock_updater = MagicMock()
                        mock_updater_cls.return_value = mock_updater
                        mock_updater.failed = AsyncMock()

                        await executor.execute(context, event_queue)

                        mock_updater.failed.assert_awaited_once()
