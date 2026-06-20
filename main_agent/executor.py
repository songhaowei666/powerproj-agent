"""业务 Agent 执行器 - A2A JSON-RPC 客户端封装 + 重试逻辑。"""

import asyncio
from typing import Dict, Any, List, Optional
from uuid import uuid4

import google.protobuf.json_format as json_format
import httpx
from a2a.types import Role
from a2a.types.a2a_pb2 import SendMessageConfiguration, SendMessageRequest

from a2a_message_parser import build_upstream_header
from intent_agent.models import SubTask
from main_agent.models import TaskOutput

MAX_RETRIES = 3
TIMEOUT_SECONDS = 60.0
A2A_HEADERS = {
    "Content-Type": "application/json",
    "A2A-Version": "1.0",
}


def _get_card_name(card: Any) -> str:
    """从 AgentCard 对象或 dict 中读取 name。"""
    if isinstance(card, dict):
        return str(card.get("name", "")).strip()
    return str(getattr(card, "name", "")).strip()


def _find_agent_url(agent_cards: List[Any], agent_name: str) -> str:
    """根据 required_agent 查找对应 Agent 的 JSON-RPC endpoint。

    Args:
        agent_cards: 可用 AgentCard 列表
        agent_name: 子任务目标 Agent 名称（AgentCard.name）

    Returns:
        A2A endpoint URL

    Raises:
        ValueError: 找不到匹配 Agent 或 endpoint 时
    """
    normalized = agent_name.strip()
    for card in agent_cards:
        if _get_card_name(card) != normalized:
            continue
        interfaces = getattr(card, "supported_interfaces", [])
        if isinstance(card, dict):
            interfaces = card.get("supported_interfaces", interfaces)
        for iface in interfaces:
            binding = getattr(iface, "protocol_binding", "")
            if isinstance(iface, dict):
                binding = iface.get("protocol_binding", binding)
            if binding.upper() == "JSONRPC":
                url = getattr(iface, "url", "")
                if isinstance(iface, dict):
                    url = iface.get("url", url)
                if url:
                    return url.rstrip("/")
        raise ValueError(
            f"Agent '{normalized}' 未配置 JSONRPC endpoint"
        )

    registered = sorted({_get_card_name(card) for card in agent_cards if _get_card_name(card)})
    raise ValueError(
        f"未找到业务 Agent '{normalized}'，"
        f"已注册 Agent: {registered}"
    )


def _find_agent_info(agent_cards: List[Any], agent_name: str) -> tuple[str, str]:
    """根据 required_agent 查找 Agent 名称与 endpoint。

    Returns:
        (agent_name, endpoint_url)
    """
    normalized = agent_name.strip()
    url = _find_agent_url(agent_cards, normalized)
    return normalized, url


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


def extract_artifact_text(artifact: Dict[str, Any]) -> str:
    """从 artifact 中提取文本（兼容 A2A parts 嵌套与内部扁平格式）。"""
    parts = _artifact_to_message_parts(artifact)
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "\n".join(texts)


def build_task_parts(
    subtask: SubTask,
    task_outputs: Dict[str, TaskOutput],
    subtask_map: Dict[str, SubTask],
    user_attachments: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """构建发送给业务 Agent 的 message.parts，注入前置依赖任务结果。

    Args:
        subtask: 当前待执行的子任务
        task_outputs: 已完成任务的输出，key 为 task_id
        subtask_map: 全部子任务定义，用于补充前置任务描述
        user_attachments: 用户消息附带的 raw/url parts

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
                    dep_id, output.required_agent, dep_name
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

    if user_attachments:
        parts.extend(user_attachments)

    return parts


def _extract_task_from_rpc_result(rpc_body: Dict[str, Any]) -> Dict[str, Any]:
    """从 JSON-RPC 响应中提取 Task 对象。"""
    rpc_result = rpc_body.get("result", {})
    return rpc_result.get("task", rpc_result)


def _extract_parts_text(parts: List[Dict[str, Any]]) -> str:
    """从 message/artifact parts 中提取文本。"""
    lines: List[str] = []
    for part in parts:
        text = part.get("text", "")
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_artifacts_from_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从业务 Agent 返回的 Task 中提取 artifacts。"""
    artifacts = task.get("artifacts", [])
    if artifacts:
        return artifacts

    status_message = task.get("status", {}).get("message", {})
    status_parts = status_message.get("parts", [])
    text = _extract_parts_text(status_parts)
    if text:
        return [{"parts": [{"text": text}]}]
    return []


def _normalize_task_state(task_state: Any) -> str:
    """将 A2A Task 状态规范化为简短名称。"""
    if not isinstance(task_state, str):
        return str(task_state)
    return task_state.replace("TASK_STATE_", "").lower().replace("_", "-")


def _is_input_required_state(task_state: Any) -> bool:
    """判断业务 Agent 是否返回 input-required。"""
    normalized = _normalize_task_state(task_state)
    return normalized in ("input-required", "inputrequired")


def _extract_status_parts(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 Task 状态消息中提取 parts 字典列表。"""
    status_message = task.get("status", {}).get("message", {})
    parts = status_message.get("parts", [])
    return parts if isinstance(parts, list) else []


def _build_send_message_payload(
    message_parts: List[Dict[str, Any]],
    business_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 A2A SendMessage JSON-RPC 请求体。"""
    req = SendMessageRequest()
    if business_task_id:
        req.message.task_id = business_task_id
    req.message.message_id = uuid4().hex
    req.message.role = Role.ROLE_USER
    for part in message_parts:
        if part.get("text"):
            req.message.parts.add().text = part["text"]
            continue
        url = part.get("url", "")
        if url:
            proto_part = req.message.parts.add()
            proto_part.url = url
            proto_part.filename = part.get("filename") or part.get("name") or "文件"
            continue
        if part.get("raw") is not None:
            import base64

            proto_part = req.message.parts.add()
            raw = part["raw"]
            if isinstance(raw, str):
                try:
                    proto_part.raw = base64.b64decode(raw)
                except Exception:
                    proto_part.raw = raw.encode("utf-8")
            else:
                proto_part.raw = bytes(raw)
            proto_part.filename = part.get("filename") or part.get("name") or "unnamed"
            media_type = part.get("mediaType") or part.get("media_type") or ""
            if media_type:
                proto_part.media_type = media_type
            continue
    req.configuration.CopyFrom(SendMessageConfiguration())
    return {
        "jsonrpc": "2.0",
        "method": "SendMessage",
        "params": json_format.MessageToDict(req),
        "id": 1,
    }


async def call_business_agent(
    subtask: SubTask,
    agent_cards: List[Any],
    session_id: str,
    task_outputs: Optional[Dict[str, TaskOutput]] = None,
    subtask_map: Optional[Dict[str, SubTask]] = None,
    business_task_id: Optional[str] = None,
    resume_text: Optional[str] = None,
    user_attachments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """调用下游业务 Agent 的 A2A JSON-RPC 接口。

    Args:
        subtask: 子任务规划
        agent_cards: 当前可用的 AgentCard 列表
        session_id: 会话ID
        task_outputs: 已完成任务输出，用于向有依赖的后置任务注入前置结果
        subtask_map: 全部子任务定义，用于构建前置结果上下文
        business_task_id: 业务 Agent 侧 task_id，resume 时复用
        resume_text: 用户补充/确认文本，非空时仅发送该文本

    Returns:
        success: {"status": "success", "artifacts": [...]}
        input_required: {"status": "input_required", "question", "parts", "business_task_id"}

    Raises:
        ValueError: 找不到匹配 Agent 或 endpoint 时
        Exception: 超过最大重试次数仍失败时抛出
    """
    url = _find_agent_url(agent_cards, subtask.required_agent)
    agent_name, _ = _find_agent_info(agent_cards, subtask.required_agent)
    if resume_text is not None:
        if not business_task_id:
            raise ValueError(
                f"恢复业务 Agent 任务 {subtask.id} 时缺少 business_task_id"
            )
        message_parts = [{"text": resume_text}]
        request_task_id = business_task_id
    else:
        message_parts = build_task_parts(
            subtask,
            task_outputs or {},
            subtask_map or {subtask.id: subtask},
            user_attachments=user_attachments,
        )
        # 首次调用不传 task_id，由业务 Agent 创建任务；续聊时才携带已有 id
        request_task_id = business_task_id
    payload = _build_send_message_payload(message_parts, request_task_id)

    last_error: Exception = Exception("Unknown error")

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{url}/",
                    json=payload,
                    headers=A2A_HEADERS,
                )
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"Agent JSON-RPC error: {data['error']}")

                task = _extract_task_from_rpc_result(data)
                returned_task_id = task.get("id") or business_task_id or ""
                task_state = task.get("status", {}).get("state", "")
                status_parts = _extract_status_parts(task)

                if task_state == "TASK_STATE_FAILED":
                    error_text = _extract_parts_text(status_parts) or "业务 Agent 执行失败"
                    raise RuntimeError(error_text)

                if _is_input_required_state(task_state):
                    question = _extract_parts_text(status_parts) or "请补充信息"
                    return {
                        "status": "input_required",
                        "question": question,
                        "parts": status_parts,
                        "business_task_id": returned_task_id,
                        "trace": {
                            "agent_name": agent_name,
                            "endpoint": url,
                            "agent_name": agent_name,
                            "required_agent": subtask.required_agent,
                            "subtask": subtask.model_dump(),
                            "message_parts": message_parts,
                            "request": payload,
                        },
                    }

                artifacts = _extract_artifacts_from_task(task)
                return {
                    "status": "success",
                    "artifacts": artifacts,
                    "business_task_id": returned_task_id,
                    "trace": {
                        "agent_name": agent_name,
                        "endpoint": url,
                        "required_agent": subtask.required_agent,
                        "subtask": subtask.model_dump(),
                        "message_parts": message_parts,
                        "request": payload,
                    },
                }

        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                break
            # 线性退避: 1s, 2s
            await asyncio.sleep(1.0 * (attempt + 1))

    raise last_error
