"""IntentAgent 封装类，对外统一入口。"""

from typing import Any, Dict, List, Sequence

from langchain_core.language_models import BaseChatModel

from intent_agent.graph import build_intent_graph
from intent_agent.models import IntentResult


def _serialize_agent_cards(agent_cards: Sequence) -> List[Dict[str, Any]]:
    """将 AgentCard 转为可序列化 dict，供 LangGraph checkpoint 使用。"""
    serialized: List[Dict[str, Any]] = []
    for card in agent_cards:
        skills: List[Dict[str, Any]] = []
        for skill in getattr(card, "skills", []):
            skills.append(
                {
                    "id": getattr(skill, "id", ""),
                    "name": getattr(skill, "name", ""),
                    "description": getattr(skill, "description", ""),
                    "tags": list(getattr(skill, "tags", [])),
                    "examples": list(getattr(skill, "examples", [])),
                }
            )
        serialized.append(
            {
                "name": getattr(card, "name", ""),
                "skills": skills,
            }
        )
    return serialized


class IntentAgent:
    """意图识别 Agent。

    基于 LangGraph 实现，支持多意图识别，基于可用 AgentCard 能力输出
    带依赖关系的任务规划列表。

    Usage:
        >>> from providers.llm_provider import get_llm
        >>> agent = IntentAgent(get_llm())
        >>> result = await agent.recognize(
        ...     "帮我统计今年的投资收益",
        ...     agent_cards=[statistics_card, investment_card],
        ... )
    """

    def __init__(self, llm: BaseChatModel):
        self._graph = build_intent_graph(llm)

    async def recognize(self, query: str, agent_cards: Sequence) -> IntentResult:
        """识别用户 query 的意图，返回任务规划列表。

        Args:
            query: 用户输入的自然语言查询
            agent_cards: 可用业务 Agent 的 AgentCard 对象序列

        Returns:
            IntentResult，包含任务目标、子任务列表、执行顺序和推理说明
        """
        state = await self._graph.ainvoke(
            {
                "query": query,
                "agent_cards": _serialize_agent_cards(agent_cards),
            }
        )
        return state["result"]
