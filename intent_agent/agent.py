"""IntentAgent 封装类，对外统一入口。"""

from langchain_core.language_models import BaseChatModel

from intent_agent.graph import build_intent_graph
from intent_agent.models import IntentResult


class IntentAgent:
    """意图识别 Agent。

    基于 LangGraph 实现，支持多意图识别，输出带依赖关系的任务规划列表。

    Usage:
        >>> from providers.llm_provider import get_llm
        >>> agent = IntentAgent(get_llm())
        >>> result = await agent.recognize("帮我统计今年的投资收益")
    """

    def __init__(self, llm: BaseChatModel):
        self._graph = build_intent_graph(llm)

    async def recognize(self, query: str) -> IntentResult:
        """识别用户 query 的意图，返回任务规划列表。

        Args:
            query: 用户输入的自然语言查询

        Returns:
            IntentResult，包含任务列表和推理说明
        """
        state = await self._graph.ainvoke({"query": query})
        return state["result"]
