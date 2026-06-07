"""业务 Agent 执行器 - A2A JSON-RPC 客户端封装 + 重试逻辑。"""

import asyncio
from typing import Dict, Any

import httpx

from intent_agent.models import TaskPlan
from main_agent.registry import get_agent_url

MAX_RETRIES = 3
TIMEOUT_SECONDS = 60.0


async def call_business_agent(task_plan: TaskPlan, session_id: str) -> Dict[str, Any]:
    """调用下游业务 Agent 的 A2A JSON-RPC 接口。

    Args:
        task_plan: 任务规划
        session_id: 会话ID

    Returns:
        {"status": "success", "artifacts": [...]}

    Raises:
        Exception: 超过最大重试次数仍失败时抛出
    """
    url = get_agent_url(task_plan.business)
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "id": f"{session_id}-{task_plan.task_id}",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_plan.description}],
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
