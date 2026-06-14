"""业务 Agent 执行器 - A2A JSON-RPC 客户端封装 + 重试逻辑。"""

import asyncio
from typing import Dict, Any, List

import httpx

from intent_agent.models import SubTask

MAX_RETRIES = 3
TIMEOUT_SECONDS = 60.0


def _find_agent_url(agent_cards: List[Any], capability: str) -> str:
    """根据 required_capability 查找对应 Agent 的 JSON-RPC endpoint。

    Args:
        agent_cards: 可用 AgentCard 列表
        capability: 子任务所需的 skill id

    Returns:
        A2A endpoint URL

    Raises:
        ValueError: 找不到匹配 capability 或 endpoint 时
    """
    for card in agent_cards:
        skills = getattr(card, "skills", [])
        for skill in skills:
            skill_id = getattr(skill, "id", "")
            if skill_id == capability:
                interfaces = getattr(card, "supported_interfaces", [])
                for iface in interfaces:
                    binding = getattr(iface, "protocol_binding", "")
                    if binding.upper() == "JSONRPC":
                        url = getattr(iface, "url", "")
                        if url:
                            return url.rstrip("/")
                raise ValueError(
                    f"Skill '{capability}' 所属 Agent 未配置 JSONRPC endpoint"
                )

    raise ValueError(
        f"未找到支持能力 '{capability}' 的 Agent，"
        f"已注册能力: {[getattr(s, 'id', '') for c in agent_cards for s in getattr(c, 'skills', [])]}"
    )


async def call_business_agent(
    subtask: SubTask,
    agent_cards: List[Any],
    session_id: str,
) -> Dict[str, Any]:
    """调用下游业务 Agent 的 A2A JSON-RPC 接口。

    Args:
        subtask: 子任务规划
        agent_cards: 当前可用的 AgentCard 列表
        session_id: 会话ID

    Returns:
        {"status": "success", "artifacts": [...]}

    Raises:
        ValueError: 找不到匹配 Agent 或 endpoint 时
        Exception: 超过最大重试次数仍失败时抛出
    """
    url = _find_agent_url(agent_cards, subtask.required_capability)
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "id": f"{session_id}-{subtask.id}",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": subtask.description}],
            },
        },
        "id": 1,
    }

    last_error: Exception = Exception("Unknown error")

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                resp = await client.post(f"{url}/", json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"Agent JSON-RPC error: {data['error']}")

                result_task = data.get("result", {})
                artifacts = result_task.get("artifacts", [])
                return {"status": "success", "artifacts": artifacts}

        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break
            # 线性退避: 1s, 2s
            await asyncio.sleep(1.0 * (attempt + 1))

    raise last_error
