"""意图识别 Agent 单元测试。"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from intent_agent.models import TaskPlan, IntentResult
from intent_agent.prompts import build_system_prompt
from intent_agent.graph import IntentState, retrieve_few_shots_node, plan_tasks_node


class TestModels:
    """Pydantic 模型序列化/反序列化测试。"""

    def test_task_plan_serialization(self):
        task = TaskPlan(
            task_id="task_1",
            business="统计业务",
            confidence=0.92,
            dependencies=[],
            description="统计今年收益",
        )
        data = task.model_dump()
        assert data["task_id"] == "task_1"
        assert data["business"] == "统计业务"
        assert data["confidence"] == 0.92

    def test_intent_result_serialization(self):
        result = IntentResult(
            tasks=[
                TaskPlan(
                    task_id="task_1",
                    business="统计业务",
                    confidence=0.92,
                    dependencies=[],
                    description="统计今年收益",
                )
            ],
            reasoning="query 涉及统计",
        )
        data = result.model_dump()
        assert len(data["tasks"]) == 1
        assert data["reasoning"] == "query 涉及统计"

    def test_intent_result_from_json(self):
        raw = {
            "tasks": [
                {
                    "task_id": "task_1",
                    "business": "规划业务",
                    "confidence": 0.85,
                    "dependencies": [],
                    "description": "制定年度计划",
                }
            ],
            "reasoning": "用户要求规划",
        }
        result = IntentResult.model_validate(raw)
        assert result.tasks[0].business == "规划业务"


class TestRagStub:
    """RAG stub 测试。"""

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_list(self):
        from intent_agent.rag_stub import retrieve_similar_examples

        result = await retrieve_similar_examples("测试 query")
        assert result == []


class TestPrompts:
    """Prompt 构建逻辑测试。"""

    def test_build_system_prompt_without_few_shots(self):
        prompt = build_system_prompt([])
        assert "多意图识别专家" in prompt
        assert "统计业务" in prompt
        assert "规划业务" in prompt
        assert "投资业务" in prompt
        assert "task_id" in prompt

    def test_build_system_prompt_with_few_shots(self):
        few_shots = [
            {
                "query": "统计一下上季度的销售数据",
                "tasks": [
                    {
                        "task_id": "task_1",
                        "business": "统计业务",
                        "confidence": 0.95,
                        "dependencies": [],
                        "description": "统计上季度销售数据",
                    }
                ],
            }
        ]
        prompt = build_system_prompt(few_shots)
        assert "少样本示例" in prompt
        assert "统计一下上季度的销售数据" in prompt
        assert "task_1" in prompt


class TestGraph:
    """LangGraph 节点链路测试。"""

    @pytest.mark.asyncio
    async def test_retrieve_few_shots_node(self):
        state: IntentState = {"query": "测试", "few_shots": [], "result": None}
        new_state = await retrieve_few_shots_node(state)
        assert new_state["query"] == "测试"
        assert new_state["few_shots"] == []
        assert new_state["result"] is None

    @pytest.mark.asyncio
    async def test_plan_tasks_node(self):
        mock_result = IntentResult(
            tasks=[
                TaskPlan(
                    task_id="task_1",
                    business="统计业务",
                    confidence=0.9,
                    dependencies=[],
                    description="测试任务",
                )
            ],
            reasoning="测试推理",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )

        state: IntentState = {
            "query": "帮我统计一下",
            "few_shots": [],
            "result": None,
        }
        new_state = await plan_tasks_node(state, mock_llm)

        assert new_state["result"] == mock_result
        mock_llm.with_structured_output.assert_called_once()


class TestAgent:
    """IntentAgent 集成测试（使用 mock LLM）。"""

    @pytest.mark.asyncio
    async def test_recognize(self):
        from intent_agent.agent import IntentAgent

        mock_result = IntentResult(
            tasks=[
                TaskPlan(
                    task_id="task_1",
                    business="投资业务",
                    confidence=0.88,
                    dependencies=[],
                    description="分析投资组合",
                )
            ],
            reasoning="测试",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )

        agent = IntentAgent(mock_llm)
        result = await agent.recognize("分析一下我的投资组合")

        assert isinstance(result, IntentResult)
        assert len(result.tasks) == 1
        assert result.tasks[0].business == "投资业务"
