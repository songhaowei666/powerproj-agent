"""意图识别 Agent 单元测试。"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from intent_agent.models import SubTask, IntentResult
from intent_agent.prompts import build_system_prompt
from intent_agent.graph import IntentState, retrieve_few_shots_node, plan_tasks_node


class TestModels:
    """Pydantic 模型序列化/反序列化测试。"""

    def test_subtask_serialization(self):
        task = SubTask(
            id="task_1",
            name="查询项目信息",
            description="查询北京西500千伏项目的基本信息",
            dependencies=[],
            expected_output="项目基本信息",
            required_capability="project-query",
            confidence=0.92,
        )
        data = task.model_dump()
        assert data["id"] == "task_1"
        assert data["required_capability"] == "project-query"
        assert data["confidence"] == 0.92

    def test_intent_result_serialization(self):
        result = IntentResult(
            task_goal="查询项目信息并下载文件",
            subtasks=[
                SubTask(
                    id="task_1",
                    name="查询项目信息",
                    description="查询北京西500千伏项目的基本信息",
                    dependencies=[],
                    expected_output="项目基本信息",
                    required_capability="project-query",
                    confidence=0.92,
                )
            ],
            execution_order=["task_1"],
            reasoning="用户要求查询项目信息",
        )
        data = result.model_dump()
        assert data["task_goal"] == "查询项目信息并下载文件"
        assert len(data["subtasks"]) == 1
        assert data["execution_order"] == ["task_1"]
        assert data["reasoning"] == "用户要求查询项目信息"

    def test_intent_result_from_json(self):
        raw = {
            "task_goal": "查询项目",
            "subtasks": [
                {
                    "id": "task_1",
                    "name": "查询项目",
                    "description": "查询项目信息",
                    "dependencies": [],
                    "expected_output": "项目信息",
                    "required_capability": "project-query",
                    "confidence": 0.85,
                }
            ],
            "execution_order": ["task_1"],
            "reasoning": "用户要求规划",
        }
        result = IntentResult.model_validate(raw)
        assert result.subtasks[0].required_capability == "project-query"


class TestRagStub:
    """RAG stub 测试。"""

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_list(self):
        from intent_agent.rag_stub import retrieve_similar_examples

        result = await retrieve_similar_examples("测试 query")
        assert result == []


class TestPrompts:
    """Prompt 构建逻辑测试。"""

    def _build_mock_agent_card(self, skill_id: str):
        """构造一个 mock AgentCard 对象。"""
        skill = MagicMock()
        skill.id = skill_id
        skill.name = "测试能力"
        skill.description = "测试能力描述"
        skill.tags = ["test"]
        skill.examples = ["示例1"]

        card = MagicMock()
        card.name = "test-agent"
        card.skills = [skill]
        return card

    def test_build_system_prompt_with_capabilities(self):
        agent_cards = [self._build_mock_agent_card("test-skill")]
        prompt = build_system_prompt([], agent_cards)
        assert "多意图识别与任务规划专家" in prompt
        assert "test-skill" in prompt
        assert "测试能力" in prompt
        assert "project-query" not in prompt

    def test_build_system_prompt_with_few_shots(self):
        agent_cards = [self._build_mock_agent_card("test-skill")]
        few_shots = [
            {
                "query": "统计一下上季度的销售数据",
                "tasks": [
                    {
                        "id": "task_1",
                        "name": "统计销售数据",
                        "description": "统计上季度销售数据",
                        "dependencies": [],
                        "expected_output": "销售数据报表",
                        "required_capability": "test-skill",
                        "confidence": 0.95,
                    }
                ],
            }
        ]
        prompt = build_system_prompt(few_shots, agent_cards)
        assert "少样本示例" in prompt
        assert "统计一下上季度的销售数据" in prompt
        assert "task_1" in prompt


class TestGraph:
    """LangGraph 节点链路测试。"""

    @pytest.mark.asyncio
    async def test_retrieve_few_shots_node(self):
        state: IntentState = {
            "query": "测试",
            "agent_cards": [],
            "few_shots": [],
            "result": None,
        }
        new_state = await retrieve_few_shots_node(state)
        assert new_state["query"] == "测试"
        assert new_state["agent_cards"] == []
        assert new_state["few_shots"] == []
        assert new_state["result"] is None

    @pytest.mark.asyncio
    async def test_plan_tasks_node(self):
        mock_result = IntentResult(
            task_goal="测试",
            subtasks=[
                SubTask(
                    id="task_1",
                    name="测试任务",
                    description="测试任务描述",
                    dependencies=[],
                    expected_output="测试结果",
                    required_capability="test-skill",
                    confidence=0.9,
                )
            ],
            execution_order=["task_1"],
            reasoning="测试推理",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )

        state: IntentState = {
            "query": "帮我查一下项目",
            "agent_cards": [],
            "few_shots": [],
            "result": None,
        }
        new_state = await plan_tasks_node(state, mock_llm)

        assert new_state["result"] == mock_result
        mock_llm.with_structured_output.assert_called_once()


class TestAgent:
    """IntentAgent 集成测试（使用 mock LLM）."""

    @pytest.mark.asyncio
    async def test_recognize(self):
        from intent_agent.agent import IntentAgent

        mock_result = IntentResult(
            task_goal="测试",
            subtasks=[
                SubTask(
                    id="task_1",
                    name="测试任务",
                    description="测试任务描述",
                    dependencies=[],
                    expected_output="测试结果",
                    required_capability="test-skill",
                    confidence=0.88,
                )
            ],
            execution_order=["task_1"],
            reasoning="测试",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )

        agent = IntentAgent(mock_llm)
        result = await agent.recognize("帮我查一下项目", agent_cards=[])

        assert isinstance(result, IntentResult)
        assert len(result.subtasks) == 1
        assert result.subtasks[0].required_capability == "test-skill"
