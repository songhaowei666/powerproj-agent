"""业务 Agent 注册表 - business 类型到 A2A endpoint 的映射。"""

from typing import Dict

AGENT_REGISTRY: Dict[str, str] = {
    "统计业务": "http://localhost:8003",
    "规划业务": "http://localhost:8001",
    "投资业务": "http://localhost:8002",
}


def get_agent_url(business: str) -> str:
    """根据业务类型获取 Agent endpoint URL。

    Args:
        business: 业务类型名称

    Returns:
        A2A endpoint URL

    Raises:
        ValueError: 未注册的业务类型
    """
    url = AGENT_REGISTRY.get(business)
    if not url:
        raise ValueError(f"未注册的业务类型: {business}，已注册: {list(AGENT_REGISTRY.keys())}")
    return url
