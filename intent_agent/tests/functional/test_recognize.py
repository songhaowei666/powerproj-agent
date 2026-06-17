"""意图识别 Agent 功能测试。

使用真实 LLM 和真实 AgentCard，站在用户角度验证意图识别能力。
不访问外部业务 Agent，仅验证任务规划结构与能力匹配。
"""

import pytest

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentInterface

from intent_agent.agent import IntentAgent
from providers.llm_provider import get_llm
from config import settings


pytestmark = pytest.mark.asyncio


def _has_llm_config() -> bool:
    """检查是否配置了 LLM 调用所需参数。"""
    return bool(settings.openai_api_key or settings.openai_api_base)


def _build_planning_card() -> AgentCard:
    """构造 planning-agent 的 AgentCard。"""
    return AgentCard(
        name="planning-agent",
        description="规划业务 Agent，负责电力项目信息查询、节点文件管理",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="project-query",
                name="项目信息查询",
                description="根据自然语言查询电力项目基本信息及聚合统计",
                tags=["planning", "project", "query"],
                examples=[
                    "查一下北京西500千伏项目的信息",
                    "所有项目变电容量的总和是多少",
                ],
            ),
            AgentSkill(
                id="file-management",
                name="节点文件管理",
                description="按节点编码上传、下载、删除文件",
                tags=["planning", "file", "upload", "download"],
                examples=["下载北京西项目的可研设计文件"],
            ),
        ],
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url="http://localhost:8001")
        ],
    )


def _build_statistics_card() -> AgentCard:
    """构造 statistics-agent 的 AgentCard。"""
    return AgentCard(
        name="statistics-agent",
        description="统计业务 Agent，负责数据分析、报表生成、指标统计等",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="data-analysis",
                name="数据分析",
                description="对业务数据进行描述性统计和趋势分析",
                tags=["statistics", "analysis"],
                examples=["分析上月销售数据"],
            ),
            AgentSkill(
                id="report-generation",
                name="报表生成",
                description="根据数据自动生成统计报表",
                tags=["statistics", "report"],
                examples=["生成月度运营报表"],
            ),
        ],
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url="http://localhost:8003")
        ],
    )


@pytest.mark.skipif(not _has_llm_config(), reason="未配置 LLM 环境变量")
async def test_recognize_planning_query():
    """测试规划业务 query 的意图识别。"""
    agent = IntentAgent(get_llm())
    agent_cards = [_build_planning_card()]

    result = await agent.recognize(
        "查一下北京西500千伏项目的信息，并下载可研设计文件",
        agent_cards=agent_cards,
    )

    assert result.task_goal
    assert result.subtasks
    assert result.execution_order
    assert result.reasoning

    # 所有子任务的目标 Agent 必须在可用列表中
    available_agents = {card.name for card in agent_cards}
    for subtask in result.subtasks:
        assert subtask.required_agent in available_agents
        assert subtask.name
        assert subtask.description
        assert subtask.expected_output

    # 执行顺序必须满足依赖关系
    order_index = {tid: idx for idx, tid in enumerate(result.execution_order)}
    for subtask in result.subtasks:
        for dep in subtask.dependencies:
            assert dep in order_index
            assert order_index[dep] < order_index[subtask.id]


@pytest.mark.skipif(not _has_llm_config(), reason="未配置 LLM 环境变量")
async def test_recognize_multi_agent_query():
    """测试跨 Agent 协作 query 的意图识别。"""
    agent = IntentAgent(get_llm())
    agent_cards = [_build_planning_card(), _build_statistics_card()]

    result = await agent.recognize(
        "查一下北京西500千伏项目的信息，并生成项目信息简报",
        agent_cards=agent_cards,
    )

    assert result.task_goal
    assert len(result.subtasks) >= 1
    assert result.execution_order

    available_agents = {card.name for card in agent_cards}
    for subtask in result.subtasks:
        assert subtask.required_agent in available_agents

    order_index = {tid: idx for idx, tid in enumerate(result.execution_order)}
    for subtask in result.subtasks:
        for dep in subtask.dependencies:
            assert dep in order_index
            assert order_index[dep] < order_index[subtask.id]


@pytest.mark.skipif(not _has_llm_config(), reason="未配置 LLM 环境变量")
async def test_recognize_all_projects_capacity_query(capsys):
    """测试全量项目规划变电容量聚合统计 query 的意图识别。"""
    agent = IntentAgent(get_llm())
    agent_cards = [_build_planning_card(), _build_statistics_card()]
    query = "所有项目的规划变电容量是多少"

    result = await agent.recognize(query, agent_cards=agent_cards)

    print("\n===== 意图识别结果 =====")
    print(result.model_dump_json(indent=2, ensure_ascii=False))
    print("========================\n")

    assert result.task_goal
    assert result.subtasks
    assert result.execution_order
    assert result.reasoning

    available_agents = {card.name for card in agent_cards}
    for subtask in result.subtasks:
        assert subtask.required_agent in available_agents
        assert subtask.name
        assert subtask.description
        assert subtask.expected_output

    order_index = {tid: idx for idx, tid in enumerate(result.execution_order)}
    for subtask in result.subtasks:
        for dep in subtask.dependencies:
            assert dep in order_index
            assert order_index[dep] < order_index[subtask.id]
