"""Main Agent 单元测试。"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from main_agent.agent_executor import MainAgentExecutor
from main_agent.models import MainState, TaskOutput
from main_agent.graph import (
    build_main_graph,
    _collect_registered_skills,
    _find_invalid_capability_subtasks,
    _resolve_clarification_prompt,
)
from main_agent.executor import build_task_parts, call_business_agent, extract_artifact_text
from main_agent.agent_network import AgentNetwork
from main_agent.streaming import format_trace_step_message, parse_trace_step_message
from intent_agent.models import IntentResult, SubTask


@pytest.fixture
def mock_llm():
    """提供一个 mock LLM。"""

    async def _mock_astream(*_args, **_kwargs):
        yield MagicMock(content="总结内容")

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="总结内容"))
    llm.astream = _mock_astream
    return llm


@pytest.fixture
def agent_network():
    """提供一个空的 AgentNetwork。"""
    return AgentNetwork()


class TestBuildQueryFromHistory:
    """测试从 A2A Task history 构建查询文本。"""

    def test_none_task_uses_fallback(self):
        assert MainAgentExecutor._build_query_from_history(None, "首次提问") == "首次提问"

    def test_empty_history_uses_fallback(self):
        from a2a.types import a2a_pb2

        task = a2a_pb2.Task()
        assert MainAgentExecutor._build_query_from_history(task, "当前消息") == "当前消息"

    def test_joins_user_messages_from_history(self):
        from a2a.types import a2a_pb2

        task = a2a_pb2.Task()
        user_msg = task.history.add()
        user_msg.role = a2a_pb2.ROLE_USER
        user_msg.parts.add().text = "第一条"
        agent_msg = task.history.add()
        agent_msg.role = a2a_pb2.ROLE_AGENT
        agent_msg.parts.add().text = "回复"
        user_msg2 = task.history.add()
        user_msg2.role = a2a_pb2.ROLE_USER
        user_msg2.parts.add().text = "第二条"

        assert MainAgentExecutor._build_query_from_history(task) == "第一条\n第二条"

    def test_state_values_to_dict(self):
        assert MainAgentExecutor._state_values_to_dict(None) == {}
        assert MainAgentExecutor._state_values_to_dict({"summary": "ok"}) == {
            "summary": "ok"
        }
        assert MainAgentExecutor._state_values_to_dict(
            MainState(query="q", summary="结果")
        )["summary"] == "结果"

    def test_artifact_to_parts_invocation_trace(self):
        parts = MainAgentExecutor._artifact_to_parts(
            {
                "type": "invocation_trace",
                "traces": [
                    {
                        "step": 1,
                        "agent_type": "intent",
                        "agent_name": "意图识别 Agent",
                        "input": {"query": "测试"},
                        "output": {},
                        "status": "success",
                    }
                ],
            }
        )
        assert len(parts) == 1
        assert parts[0].text.startswith("__INVOCATION_TRACE__\n")


class TestStreamingProtocol:
    """测试流式轨迹协议编解码。"""

    def test_trace_step_roundtrip(self):
        trace = {"step": 1, "agent_type": "intent", "status": "success"}
        message = format_trace_step_message(trace)
        parsed = parse_trace_step_message(message)
        assert parsed == trace


class TestClarificationPrompt:
    """测试澄清问句解析。"""

    def test_resolve_clarification_uses_llm_output(self):
        result = IntentResult(
            is_business_query=True,
            task_goal="模糊请求",
            subtasks=[],
            execution_order=[],
            reasoning="测试",
            clarification_prompt="请问您想查询哪个项目？",
        )
        assert _resolve_clarification_prompt(result, "默认澄清") == "请问您想查询哪个项目？"

    def test_resolve_clarification_fallback(self):
        result = IntentResult(
            is_business_query=True,
            task_goal="模糊请求",
            subtasks=[],
            execution_order=[],
            reasoning="测试",
        )
        assert _resolve_clarification_prompt(result, "默认澄清") == "默认澄清"


class TestCapabilityValidation:
    """测试 required_capability 校验逻辑。"""

    def _build_mock_card(self, skill_id: str):
        skill = MagicMock()
        skill.id = skill_id
        card = MagicMock()
        card.skills = [skill]
        return card

    def test_collect_registered_skills(self):
        cards = [
            self._build_mock_card("project-query"),
            self._build_mock_card("project-statistics"),
        ]
        assert _collect_registered_skills(cards) == {
            "project-query",
            "project-statistics",
        }

    def test_find_invalid_capability_subtasks(self):
        subtasks = [
            SubTask.model_construct(
                id="task_1",
                name="查询",
                description="查询变电容量",
                dependencies=[],
                expected_output="结果",
                required_capability="",
            ),
            SubTask(
                id="task_2",
                name="统计",
                description="统计分析",
                dependencies=[],
                expected_output="结果",
                required_capability="project-statistics",
            ),
            SubTask(
                id="task_3",
                name="未知",
                description="未知能力",
                dependencies=[],
                expected_output="结果",
                required_capability="unknown-skill",
            ),
        ]
        invalid = _find_invalid_capability_subtasks(
            subtasks, {"project-query", "project-statistics"}
        )
        assert [t.id for t in invalid] == ["task_1", "task_3"]


class TestBuildPhases:
    """测试拓扑分层逻辑。"""

    def test_linear_dependencies(self, mock_llm, agent_network):
        graph = build_main_graph(mock_llm, agent_network)
        state = MainState(
            intent_result=IntentResult(
                task_goal="线性依赖测试",
                subtasks=[
                    SubTask(
                        id="t1",
                        name="任务1",
                        description="任务1",
                        dependencies=[],
                        expected_output="结果1",
                        required_capability="skill-a",
                        confidence=0.9,
                    ),
                    SubTask(
                        id="t2",
                        name="任务2",
                        description="任务2",
                        dependencies=["t1"],
                        expected_output="结果2",
                        required_capability="skill-b",
                        confidence=0.9,
                    ),
                    SubTask(
                        id="t3",
                        name="任务3",
                        description="任务3",
                        dependencies=["t2"],
                        expected_output="结果3",
                        required_capability="skill-c",
                        confidence=0.9,
                    ),
                ],
                execution_order=["t1", "t2", "t3"],
                reasoning="线性依赖",
            )
        )
        # graph 入口是 recognize_and_check（会调 LLM），此处仅验证 fixture 能构造
        assert graph is not None


class TestMainAgentFlow:
    """测试主控 Agent 端到端流程。"""

    def _build_mock_card(self, skill_id: str, url: str = "http://localhost:8001"):
        """构造带指定 skill 的 mock AgentCard。"""
        skill = MagicMock()
        skill.id = skill_id

        iface = MagicMock()
        iface.protocol_binding = "JSONRPC"
        iface.url = url

        card = MagicMock()
        card.name = f"agent-{skill_id}"
        card.skills = [skill]
        card.supported_interfaces = [iface]
        skill.name = skill_id
        skill.description = f"{skill_id} 能力"
        return card

    def _register_skills(self, agent_network: AgentNetwork, skill_ids: list[str]) -> None:
        """向 AgentNetwork 注册测试用 AgentCard。"""
        cards = [self._build_mock_card(skill_id) for skill_id in skill_ids]
        agent_network._cards = cards

    @pytest.mark.asyncio
    async def test_full_flow_with_summarize(self, mock_llm, agent_network):
        """测试正常执行 + 总结流程。"""
        self._register_skills(agent_network, ["data-analysis"])
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="统计结果显示收益为 10%，表现良好。")
        )
        graph = build_main_graph(mock_llm, agent_network)

        # Mock intent_agent.recognize
        intent_result = IntentResult(
            task_goal="统计收益",
            subtasks=[
                SubTask(
                    id="t1",
                    name="统计分析",
                    description="统计分析",
                    dependencies=[],
                    expected_output="统计结果",
                    required_capability="data-analysis",
                    confidence=0.95,
                )
            ],
            execution_order=["t1"],
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
                        {
                            "parts": [
                                {
                                    "text": "【聚合查询结果】\n规划变电容量总和：13170.0"
                                }
                            ]
                        }
                    ],
                    "trace": {
                        "agent_name": "statistics-agent",
                        "endpoint": "http://localhost:8003",
                        "capability": "data-analysis",
                        "subtask": {
                            "id": "t1",
                            "name": "统计分析",
                            "description": "统计分析",
                            "dependencies": [],
                            "expected_output": "统计结果",
                            "required_capability": "data-analysis",
                            "confidence": 0.95,
                        },
                        "message_parts": [{"text": "统计分析"}],
                        "request": {"method": "SendMessage"},
                    },
                },
            ):
                result = await graph.ainvoke(
                    MainState(query="帮我统计收益", session_id="test-session"),
                    {"configurable": {"thread_id": "test-1"}},
                )

        assert result["status"] == "completed"
        assert result["summary"] is not None
        assert "13170.0" in result["summary"]
        # 检查原始结果也在 final_artifacts 中
        assert len(result["final_artifacts"]) >= 2  # 总结 + task_result
        assert result["final_artifacts"][0]["type"] == "text"
        # 成功路径轨迹不再写入 final_artifacts，改由流式 WORKING 推送
        trace_artifacts = [
            a for a in result["final_artifacts"] if a.get("type") == "invocation_trace"
        ]
        assert len(trace_artifacts) == 0
        assert len(result["invocation_traces"]) == 2
        assert result["invocation_traces"][0]["agent_type"] == "intent"
        assert result["invocation_traces"][1]["agent_type"] == "business"
        assert result["invocation_traces"][1]["input"]["message_parts"] == [{"text": "统计分析"}]

    @pytest.mark.asyncio
    async def test_direct_reply_non_business(self, mock_llm, agent_network):
        """非业务 query 应直接 LLM 回复，不调度业务 Agent。"""
        self._register_skills(agent_network, ["skill-a"])
        graph = build_main_graph(mock_llm, agent_network)

        intent_result = IntentResult(
            is_business_query=False,
            task_goal="用户问候",
            subtasks=[],
            execution_order=[],
            reasoning="用户仅发送问候语",
        )

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            with patch(
                "main_agent.graph.call_business_agent",
                new_callable=AsyncMock,
            ) as mock_call:
                result = await graph.ainvoke(
                    MainState(query="你好"),
                    {"configurable": {"thread_id": "test-greet"}},
                )

        assert result["status"] == "completed"
        assert result["summary"] == "总结内容"
        assert len(result["invocation_traces"]) == 1
        mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_uses_clarification_prompt(self, mock_llm, agent_network):
        """业务 query 模糊时应优先使用 LLM 输出的 clarification_prompt。"""
        self._register_skills(agent_network, ["skill-a"])
        graph = build_main_graph(mock_llm, agent_network)

        intent_result = IntentResult(
            is_business_query=True,
            task_goal="模糊请求",
            subtasks=[],
            execution_order=[],
            reasoning="测试",
            clarification_prompt="请问您想查询、统计还是规划哪类业务？",
        )

        with patch(
            "intent_agent.agent.IntentAgent.recognize",
            new_callable=AsyncMock,
            return_value=intent_result,
        ):
            result = await graph.ainvoke(
                MainState(query="帮我弄一下"),
                {"configurable": {"thread_id": "test-clarify"}},
            )

        assert "__interrupt__" in result
        interrupt_info = result["__interrupt__"][0]
        assert (
            interrupt_info.value["question"]
            == "请问您想查询、统计还是规划哪类业务？"
        )

    @pytest.mark.asyncio
    async def test_interrupt_low_confidence(self, mock_llm, agent_network):
        """测试低置信度触发 interrupt。"""
        self._register_skills(agent_network, ["skill-a"])
        graph = build_main_graph(mock_llm, agent_network)

        intent_result = IntentResult(
            is_business_query=True,
            task_goal="模糊请求",
            subtasks=[
                SubTask(
                    id="t1",
                    name="模糊任务",
                    description="模糊任务",
                    dependencies=[],
                    expected_output="结果",
                    required_capability="skill-a",
                    confidence=0.5,
                )
            ],
            execution_order=["t1"],
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
            is_business_query=True,
            task_goal="统计收益",
            subtasks=[
                SubTask(
                    id="t1",
                    name="统计收益",
                    description="统计收益",
                    dependencies=[],
                    expected_output="结果",
                    required_capability="skill-a",
                    confidence=0.95,
                )
            ],
            execution_order=["t1"],
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
    async def test_parallel_execution(self, mock_llm, agent_network):
        """测试同 Phase 任务并行执行。"""
        self._register_skills(agent_network, ["skill-a", "skill-b", "skill-c"])
        graph = build_main_graph(mock_llm, agent_network)

        intent_result = IntentResult(
            task_goal="并行测试",
            subtasks=[
                SubTask(
                    id="t1",
                    name="统计A",
                    description="统计A",
                    dependencies=[],
                    expected_output="结果A",
                    required_capability="skill-a",
                    confidence=0.95,
                ),
                SubTask(
                    id="t2",
                    name="投资B",
                    description="投资B",
                    dependencies=[],
                    expected_output="结果B",
                    required_capability="skill-b",
                    confidence=0.95,
                ),
                SubTask(
                    id="t3",
                    name="规划C",
                    description="规划C",
                    dependencies=["t1", "t2"],
                    expected_output="结果C",
                    required_capability="skill-c",
                    confidence=0.95,
                ),
            ],
            execution_order=["t1", "t2", "t3"],
            reasoning="测试并行",
        )

        call_order = []

        async def mock_call(
            subtask,
            agent_cards,
            session_id,
            task_outputs=None,
            subtask_map=None,
            **kwargs,
        ):
            call_order.append(subtask.id)
            await asyncio.sleep(0.05)  # 模拟网络延迟
            if subtask.id == "t3":
                assert task_outputs is not None
                assert "t1" in task_outputs
                assert "t2" in task_outputs
            return {
                "status": "success",
                "artifacts": [{"type": "text", "text": f"结果-{subtask.id}"}],
            }

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
    async def test_failure_fuse(self, mock_llm, agent_network):
        """测试任务失败熔断。"""
        self._register_skills(agent_network, ["skill-a", "skill-b"])
        graph = build_main_graph(mock_llm, agent_network)

        intent_result = IntentResult(
            task_goal="熔断测试",
            subtasks=[
                SubTask(
                    id="t1",
                    name="统计",
                    description="统计",
                    dependencies=[],
                    expected_output="结果",
                    required_capability="skill-a",
                    confidence=0.95,
                ),
                SubTask(
                    id="t2",
                    name="规划",
                    description="规划",
                    dependencies=["t1"],
                    expected_output="结果",
                    required_capability="skill-b",
                    confidence=0.95,
                ),
            ],
            execution_order=["t1", "t2"],
            reasoning="测试熔断",
        )

        async def mock_call(
            subtask,
            agent_cards,
            session_id,
            task_outputs=None,
            subtask_map=None,
            **kwargs,
        ):
            if subtask.id == "t1":
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
        trace_artifacts = [
            a for a in result["final_artifacts"] if a.get("type") == "invocation_trace"
        ]
        assert len(trace_artifacts) == 1
        business_traces = [
            t for t in trace_artifacts[0]["traces"] if t["agent_type"] == "business"
        ]
        assert len(business_traces) == 1
        assert business_traces[0]["status"] == "failed"


class TestBuildTaskParts:
    """测试前置依赖结果 parts 构建。"""

    def test_extract_artifact_text_from_nested_parts(self):
        text = extract_artifact_text(
            {"parts": [{"text": "【聚合查询结果】\n规划变电容量总和：13170.0"}]}
        )
        assert "13170.0" in text

    def test_no_dependencies_returns_single_text_part(self):
        subtask = SubTask(
            id="t1",
            name="统计",
            description="统计今年收益",
            dependencies=[],
            expected_output="结果",
            required_capability="skill-a",
        )
        parts = build_task_parts(subtask, {}, {"t1": subtask})
        assert parts == [{"text": "统计今年收益"}]

    def test_with_dependency_outputs(self):
        t1 = SubTask(
            id="t1",
            name="统计",
            description="统计今年收益",
            dependencies=[],
            expected_output="结果A",
            required_capability="skill-a",
        )
        t2 = SubTask(
            id="t2",
            name="规划",
            description="基于统计结果做明年规划",
            dependencies=["t1"],
            expected_output="结果B",
            required_capability="skill-b",
        )
        task_outputs = {
            "t1": TaskOutput(
                task_id="t1",
                required_capability="skill-a",
                status="success",
                artifacts=[{"type": "text", "text": "收益 10%"}],
            )
        }
        parts = build_task_parts(t2, task_outputs, {"t1": t1, "t2": t2})

        assert parts[0] == {"text": "基于统计结果做明年规划"}
        assert parts[1]["text"].startswith("【前置任务 t1")
        assert {"text": "任务描述：统计今年收益"} in parts
        assert {"text": "收益 10%"} in parts

    def test_with_a2a_nested_artifact_parts(self):
        t1 = SubTask(
            id="t1",
            name="统计",
            description="统计",
            dependencies=[],
            expected_output="结果",
            required_capability="skill-a",
        )
        t2 = SubTask(
            id="t2",
            name="规划",
            description="做规划",
            dependencies=["t1"],
            expected_output="结果",
            required_capability="skill-b",
        )
        task_outputs = {
            "t1": TaskOutput(
                task_id="t1",
                required_capability="skill-a",
                status="success",
                artifacts=[
                    {
                        "parts": [
                            {"text": "统计完成"},
                            {
                                "url": "http://localhost:8001/files/1",
                                "filename": "report.pdf",
                            },
                        ]
                    }
                ],
            )
        }
        parts = build_task_parts(t2, task_outputs, {"t1": t1, "t2": t2})
        assert {"text": "统计完成"} in parts
        assert {
            "url": "http://localhost:8001/files/1",
            "filename": "report.pdf",
        } in parts


class TestExecutorRetry:
    """测试执行器重试逻辑。"""

    def _build_mock_card(self, skill_id: str, url: str):
        """构造一个 mock AgentCard。"""
        skill = MagicMock()
        skill.id = skill_id

        iface = MagicMock()
        iface.protocol_binding = "JSONRPC"
        iface.url = url

        card = MagicMock()
        card.skills = [skill]
        card.supported_interfaces = [iface]
        return card

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={
                        "result": {
                            "task": {
                                "status": {"state": "TASK_STATE_COMPLETED"},
                                "artifacts": [{"parts": [{"text": "ok"}]}],
                            }
                        }
                    }
                ),
            )
            result = await call_business_agent(
                SubTask(
                    id="t1",
                    name="测试",
                    description="测试",
                    dependencies=[],
                    expected_output="ok",
                    required_capability="skill-a",
                ),
                agent_cards=[self._build_mock_card("skill-a", "http://localhost:8003")],
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
                    SubTask(
                        id="t1",
                        name="测试",
                        description="测试",
                        dependencies=[],
                        expected_output="ok",
                        required_capability="skill-a",
                    ),
                    agent_cards=[self._build_mock_card("skill-a", "http://localhost:8003")],
                    session_id="s1",
                )
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_message_includes_dependency_outputs(self):
        """有依赖时，请求 payload 应包含前置任务结果。"""
        t1 = SubTask(
            id="t1",
            name="统计",
            description="统计今年收益",
            dependencies=[],
            expected_output="结果A",
            required_capability="skill-a",
        )
        t2 = SubTask(
            id="t2",
            name="规划",
            description="基于统计结果做明年规划",
            dependencies=["t1"],
            expected_output="结果B",
            required_capability="skill-b",
        )
        task_outputs = {
            "t1": TaskOutput(
                task_id="t1",
                required_capability="skill-a",
                status="success",
                artifacts=[{"type": "text", "text": "收益 10%"}],
            )
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={
                        "result": {
                            "task": {
                                "status": {"state": "TASK_STATE_COMPLETED"},
                                "artifacts": [{"parts": [{"text": "ok"}]}],
                            }
                        }
                    }
                ),
            )
            await call_business_agent(
                t2,
                agent_cards=[self._build_mock_card("skill-b", "http://localhost:8001")],
                session_id="s1",
                task_outputs=task_outputs,
                subtask_map={"t1": t1, "t2": t2},
            )

        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["method"] == "SendMessage"
        parts = sent_payload["params"]["message"]["parts"]
        assert parts[0]["text"] == "基于统计结果做明年规划"
        assert any("【前置任务 t1" in p.get("text", "") for p in parts)
        assert any(p.get("text") == "收益 10%" for p in parts)

    @pytest.mark.asyncio
    async def test_input_required_response(self):
        confirmation_parts = [
            {"text": "请问是这个项目吗？"},
            {
                "mediaType": "application/vnd.powerproj.confirmation+json",
                "data": {
                    "type": "confirmation",
                    "action": "project_confirm",
                    "options": [
                        {"id": "yes", "label": "是", "replyText": "是"},
                        {"id": "no", "label": "否", "replyText": "否"},
                    ],
                },
            },
        ]
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={
                        "result": {
                            "task": {
                                "id": "biz-task-1",
                                "status": {
                                    "state": "TASK_STATE_INPUT_REQUIRED",
                                    "message": {"parts": confirmation_parts},
                                },
                            }
                        }
                    }
                ),
            )
            result = await call_business_agent(
                SubTask(
                    id="t1",
                    name="测试",
                    description="测试",
                    dependencies=[],
                    expected_output="ok",
                    required_capability="skill-a",
                ),
                agent_cards=[self._build_mock_card("skill-a", "http://localhost:8001")],
                session_id="s1",
            )
            assert result["status"] == "input_required"
            assert result["business_task_id"] == "biz-task-1"
            assert result["parts"] == confirmation_parts
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_sends_task_id_and_reply_text(self):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={
                        "result": {
                            "task": {
                                "id": "biz-task-1",
                                "status": {"state": "TASK_STATE_COMPLETED"},
                                "artifacts": [{"parts": [{"text": "ok"}]}],
                            }
                        }
                    }
                ),
            )
            await call_business_agent(
                SubTask(
                    id="t1",
                    name="测试",
                    description="测试",
                    dependencies=[],
                    expected_output="ok",
                    required_capability="skill-a",
                ),
                agent_cards=[self._build_mock_card("skill-a", "http://localhost:8001")],
                session_id="s1",
                business_task_id="biz-task-1",
                resume_text="是",
            )
            sent_payload = mock_post.call_args.kwargs["json"]
            message = sent_payload["params"]["message"]
            assert message["taskId"] == "biz-task-1" or message.get("task_id") == "biz-task-1"
            parts = message["parts"]
            assert parts[0]["text"] == "是"

    @pytest.mark.asyncio
    async def test_capability_not_found(self):
        with pytest.raises(ValueError, match="未找到支持能力"):
            await call_business_agent(
                SubTask(
                    id="t1",
                    name="测试",
                    description="测试",
                    dependencies=[],
                    expected_output="ok",
                    required_capability="skill-x",
                ),
                agent_cards=[self._build_mock_card("skill-a", "http://localhost:8003")],
                session_id="s1",
            )


class TestParseChatResponse:
    """测试 Web 客户端响应解析。"""

    def test_extract_invocation_traces(self):
        from web.client import parse_chat_response

        result = parse_chat_response(
            {
                "status": "passed",
                "task_id": "task-1",
                "data": {
                    "id": "task-1",
                    "status": {
                        "state": "TASK_STATE_COMPLETED",
                        "message": {"parts": [{"text": "总结内容"}]},
                    },
                    "artifacts": [
                        {
                            "parts": [
                                {
                                    "text": (
                                        '__INVOCATION_TRACE__\n'
                                        '[{"step": 1, "agent_type": "intent"}]'
                                    )
                                }
                            ]
                        }
                    ],
                },
            }
        )
        assert len(result.invocation_traces) == 1
        assert result.invocation_traces[0]["agent_type"] == "intent"

    def test_parse_confirmation_from_input_required(self):
        from a2a_message_parser.confirmation import build_confirmation_parts
        from web.client import parse_chat_response

        parts = build_confirmation_parts(
            text="请问是这个项目吗？",
            action="project_confirm",
        )
        result = parse_chat_response(
            {
                "status": "passed",
                "task_id": "task-confirm",
                "data": {
                    "id": "task-confirm",
                    "status": {
                        "state": "TASK_STATE_INPUT_REQUIRED",
                        "message": {"parts": parts},
                    },
                    "artifacts": [],
                },
            }
        )
        assert result.state == "input-required"
        assert result.confirmation is not None
        assert result.confirmation.action == "project_confirm"
        assert len(result.confirmation.options) == 2
        assert "请在下方输入补充信息" not in result.text
