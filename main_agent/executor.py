"""业务 Agent 执行器 - A2A JSON-RPC 客户端封装 + 重试逻辑。"""

import asyncio
from typing import Dict, Any, List, Optional

import httpx

from a2a_message_parser import build_upstream_header
from intent_agent.models import SubTask
from main_agent.models import TaskOutput

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


def _artifact_to_message_parts(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将单条 artifact 转换为 message parts（支持 A2A 嵌套与内部扁平格式）。"""
    nested_parts = artifact.get("parts")
    if isinstance(nested_parts, list) and nested_parts:
        return [_normalize_output_part(part) for part in nested_parts if _normalize_output_part(part)]

    artifact_type = artifact.get("type")
    if artifact_type == "text":
        text = artifact.get("text", "")
        return [{"text": text}] if text else []

    if artifact_type == "file" or "url" in artifact:
        file_info = artifact.get("file", {})
        url = artifact.get("url") or file_info.get("uri", "")
        if not url:
            return []
        filename = (
            artifact.get("name")
            or artifact.get("filename")
            or file_info.get("name")
            or "文件"
        )
        return [{"url": url, "filename": filename}]

    return []


def _normalize_output_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """归一化 artifact 内的 part 字典。"""
    if part.get("text"):
        return {"text": part["text"]}
    url = part.get("url", "")
    if url:
        return {
            "url": url,
            "filename": part.get("filename") or part.get("name") or "",
        }
    return None


def build_task_parts(
    subtask: SubTask,
    task_outputs: Dict[str, TaskOutput],
    subtask_map: Dict[str, SubTask],
) -> List[Dict[str, Any]]:
    """构建发送给业务 Agent 的 message.parts，注入前置依赖任务结果。

    Args:
        subtask: 当前待执行的子任务
        task_outputs: 已完成任务的输出，key 为 task_id
        subtask_map: 全部子任务定义，用于补充前置任务描述

    Returns:
        parts 列表，首项为当前任务 text，后续为前置任务分段及原始 parts
    """
    parts: List[Dict[str, Any]] = [{"text": subtask.description}]

    for dep_id in subtask.dependencies:
        output = task_outputs.get(dep_id)
        if not output:
            continue

        dep_subtask = subtask_map.get(dep_id)
        dep_name = dep_subtask.name if dep_subtask else dep_id
        parts.append(
            {
                "text": build_upstream_header(
                    dep_id, output.required_capability, dep_name
                )
            }
        )

        if dep_subtask and dep_subtask.description:
            parts.append({"text": f"任务描述：{dep_subtask.description}"})

        dep_parts: List[Dict[str, Any]] = []
        for artifact in output.artifacts:
            dep_parts.extend(_artifact_to_message_parts(artifact))

        if dep_parts:
            parts.extend(dep_parts)
        else:
            parts.append({"text": "（无结果）"})

    return parts


async def call_business_agent(
    subtask: SubTask,
    agent_cards: List[Any],
    session_id: str,
    task_outputs: Optional[Dict[str, TaskOutput]] = None,
    subtask_map: Optional[Dict[str, SubTask]] = None,
) -> Dict[str, Any]:
    """调用下游业务 Agent 的 A2A JSON-RPC 接口。

    Args:
        subtask: 子任务规划
        agent_cards: 当前可用的 AgentCard 列表
        session_id: 会话ID
        task_outputs: 已完成任务输出，用于向有依赖的后置任务注入前置结果
        subtask_map: 全部子任务定义，用于构建前置结果上下文

    Returns:
        {"status": "success", "artifacts": [...]}

    Raises:
        ValueError: 找不到匹配 Agent 或 endpoint 时
        Exception: 超过最大重试次数仍失败时抛出
    """
    url = _find_agent_url(agent_cards, subtask.required_capability)
    message_parts = build_task_parts(
        subtask,
        task_outputs or {},
        subtask_map or {subtask.id: subtask},
    )
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "id": f"{session_id}-{subtask.id}",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": message_parts,
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
