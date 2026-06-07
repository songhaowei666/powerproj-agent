"""Main Agent 单元测试。"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from main_agent.models import MainState, TaskOutput
from main_agent.graph import build_main_graph
from main_agent.executor import call_business_agent
from intent_agent.models import IntentResult, TaskPlan


@pytest.fixture
def mock_llm():
    """提供一个 mock LLM。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="总结内容"))
    return llm


class TestBuildPhases:
    """测试拓扑分层逻辑。"""

    def test_linear_dependencies(self, mock_llm):
        graph = build_main_graph(mock_llm)
        state = MainState(
            intent_result=IntentResult(
                tasks=[
                    TaskPlan(task_id="t1", business="统计业务", confidence=0.9, dependencies=[], description="统计"),
                    TaskPlan(task_id="t2", business="规划业务", confidence=0.9, dependencies=["t1"], description="规划"),
                    TaskPlan(task_id="t3", business="投资业务", confidence=0.9, dependencies=["t2"], description="投资"),
                ],
                reasoning="线性依赖",
            )
        )
        # 直接调用 build_phases 节点（通过 graph 走一个简化的路径来验证）
        # 由于 graph 入口是 recognize_and_check（会调 LLM），我们单独测试拓扑逻辑


class TestMainAgentFlow:
    """测试主控 Agent 端到端流程。"""

    @pytest.mark.asyncio
    async def test_full_flow_with_summarize(self, mock_llm):
        """测试正常执行 + 总结流程。"""
        # 配置 mock LLM 返回包含业务内容的总结
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="统计结果显示收益为 10%，表现良好。"))
        graph = build_main_graph(mock_llm)

        # Mock intent_agent.recognize
        intent_result = IntentResult(
            tasks=[
                TaskPlan(task_id="t1", business="统计业务", confidence=0.95, dependencies=[], description="统计分析"),
            ],
            reasoning="测试",
        )

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            with patch(
                "main_agent.graph.call_business_agent",
                new_callable=AsyncMock,
                return_value={
                    "status": "success",
                    "artifacts": [
                        {"type": "text", "text": "统计结果: 收益 10%"},
                        {"type": "file", "url": "http://example.com/report.xlsx"},
                    ],
                },
            ):
                result = await graph.ainvoke(
                    MainState(query="帮我统计收益", session_id="test-session"),
                    {"configurable": {"thread_id": "test-1"}},
                )

        assert result["status"] == "completed"
        assert result["summary"] is not None
        assert "收益" in result["summary"]
        # 检查文件链接是否被附加到总结中
        assert "report.xlsx" in result["summary"] or "example.com" in result["summary"]
        # 检查原始结果也在 final_artifacts 中
        assert len(result["final_artifacts"]) >= 2  # 总结 + task_result
        assert result["final_artifacts"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_interrupt_low_confidence(self, mock_llm):
        """测试低置信度触发 interrupt。"""
        graph = build_main_graph(mock_llm)

        intent_result = IntentResult(
            tasks=[
                TaskPlan(task_id="t1", business="统计业务", confidence=0.5, dependencies=[], description="模糊任务"),
            ],
            reasoning="测试",
        )

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            result = await graph.ainvoke(
                MainState(query="帮我弄一下"),
                {"configurable": {"thread_id": "test-2"}},
            )

        # 应该返回 interrupt 信息
        assert "__interrupt__" in result
        interrupt_info = result["__interrupt__"][0]
        assert "question" in interrupt_info.value

        # 模拟恢复：用户补充信息后重新识别（此时置信度足够）
        intent_result_ok = IntentResult(
            tasks=[
                TaskPlan(task_id="t1", business="统计业务", confidence=0.95, dependencies=[], description="统计收益"),
            ],
            reasoning="测试",
        )

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result_ok,
        ):
            with patch(
                "main_agent.graph.call_business_agent",
                new_callable=AsyncMock,
                return_value={"status": "success", "artifacts": [{"type": "text", "text": "结果"}]},
            ):
                from langgraph.types import Command
                result2 = await graph.ainvoke(
                    Command(resume="我想统计今年的投资收益"),
                    {"configurable": {"thread_id": "test-2"}},
                )

        assert result2["status"] == "completed"
        assert "模糊任务" not in result2.get("summary", "")

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_llm):
        """测试同 Phase 任务并行执行。"""
        graph = build_main_graph(mock_llm)

        intent_result = IntentResult(
            tasks=[
                TaskPlan(task_id="t1", business="统计业务", confidence=0.95, dependencies=[], description="统计A"),
                TaskPlan(task_id="t2", business="投资业务", confidence=0.95, dependencies=[], description="投资B"),
                TaskPlan(task_id="t3", business="规划业务", confidence=0.95, dependencies=["t1", "t2"], description="规划C"),
            ],
            reasoning="测试并行",
        )

        call_order = []

        async def mock_call(task_plan, session_id):
            call_order.append(task_plan.task_id)
            await asyncio.sleep(0.05)  # 模拟网络延迟
            return {"status": "success", "artifacts": [{"type": "text", "text": f"结果-{task_plan.task_id}"}]}

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            with patch("main_agent.graph.call_business_agent", side_effect=mock_call):
                result = await graph.ainvoke(
                    MainState(query="统计和投资"),
                    {"configurable": {"thread_id": "test-3"}},
                )

        assert result["status"] == "completed"
        # t1 和 t2 应该在同一阶段被调用
        phases = result["phases"]
        assert len(phases) == 2  # Phase0: [t1, t2], Phase1: [t3]
        assert set(phases[0]) == {"t1", "t2"}
        assert phases[1] == ["t3"]

    @pytest.mark.asyncio
    async def test_failure_fuse(self, mock_llm):
        """测试任务失败熔断。"""
        graph = build_main_graph(mock_llm)

        intent_result = IntentResult(
            tasks=[
                TaskPlan(task_id="t1", business="统计业务", confidence=0.95, dependencies=[], description="统计"),
                TaskPlan(task_id="t2", business="规划业务", confidence=0.95, dependencies=["t1"], description="规划"),
            ],
            reasoning="测试熔断",
        )

        async def mock_call(task_plan, session_id):
            if task_plan.task_id == "t1":
                raise RuntimeError("连接失败")
            return {"status": "success", "artifacts": []}

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            with patch("main_agent.graph.call_business_agent", side_effect=mock_call):
                result = await graph.ainvoke(
                    MainState(query="统计和规划"),
                    {"configurable": {"thread_id": "test-4"}},
                )

        assert result["status"] == "failed"
        assert result["failed_task_id"] == "t1"
        assert "连接失败" in result["error_message"]
        # t2 不应该被执行
        assert "t2" not in result.get("task_outputs", {})


class TestExecutorRetry:
    """测试执行器重试逻辑。"""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={
                    "result": {"artifacts": [{"type": "text", "text": "ok"}]}
                }),
            )
            result = await call_business_agent(
                TaskPlan(task_id="t1", business="统计业务", confidence=0.9, dependencies=[], description="测试"),
                session_id="s1",
            )
            assert result["status"] == "success"
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_fail(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = RuntimeError("连接超时")
            with pytest.raises(RuntimeError, match="连接超时"):
                await call_business_agent(
                    TaskPlan(task_id="t1", business="统计业务", confidence=0.9, dependencies=[], description="测试"),
                    session_id="s1",
                )
            assert mock_post.call_count == 3
