"""AgentNetwork - 维护可用 A2A Agent 的注册表与 AgentCard 发现。"""

from typing import List, Sequence
import asyncio
import logging

import httpx
from a2a.types import AgentCard

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
AGENT_CARD_PATH = "/.well-known/agent.json"


class AgentNetwork:
    """A2A Agent 网络管理器。

    负责注册 A2A Agent endpoint、拉取并缓存 AgentCard、向意图识别与执行模块
    提供统一的能力视图。

    Usage:
        >>> network = AgentNetwork()
        >>> network.register("http://localhost:8001")
        >>> cards = await network.discover()
        >>> print([c.name for c in cards])
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self._urls: set[str] = set()
        self._cards: List[AgentCard] = []
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    def register(self, url: str) -> None:
        """注册一个 A2A Agent endpoint。"""
        normalized = url.rstrip("/")
        self._urls.add(normalized)

    def register_from_config(self, urls: Sequence[str]) -> None:
        """从配置批量注册 endpoint。"""
        for url in urls:
            self.register(url)

    async def discover(self) -> List[AgentCard]:
        """重新拉取所有已注册 Agent 的 AgentCard。

        单个 Agent 拉取失败仅记录日志，不影响其他 Agent。

        Returns:
            成功拉取到的 AgentCard 列表
        """
        if not self._urls:
            logger.warning("AgentNetwork 中未注册任何 endpoint")
            self._cards = []
            return []

        coros = [self._fetch_card(url) for url in sorted(self._urls)]
        results = await asyncio.gather(*coros, return_exceptions=True)

        cards: List[AgentCard] = []
        for url, result in zip(sorted(self._urls), results):
            if isinstance(result, Exception):
                logger.warning(f"获取 AgentCard 失败 {url}: {result}")
                continue
            if result is not None:
                cards.append(result)

        self._cards = cards
        return cards

    async def _fetch_card(self, url: str) -> AgentCard | None:
        """从 endpoint 拉取 AgentCard。"""
        response = await self._client.get(f"{url}{AGENT_CARD_PATH}")
        response.raise_for_status()
        data = response.json()
        return AgentCard(**data)

    def get_cards(self) -> List[AgentCard]:
        """返回最近一次 discover 的缓存结果。"""
        return self._cards

    async def aclose(self) -> None:
        """关闭内部 HTTP 客户端。"""
        await self._client.aclose()
