"""主控 Agent A2A 聊天客户端。"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

import google.protobuf.json_format as json_format
import httpx
from a2a.types import Role
from a2a.types.a2a_pb2 import SendMessageConfiguration, SendMessageRequest


@dataclass
class ChatResponse:
    """解析后的聊天响应。"""

    task_id: str
    state: str
    text: str
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    invocation_traces: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


class MainAgentClient:
    """通过 A2A JSON-RPC 与主控 Agent 通信的客户端。

    Args:
        base_url: 主控 Agent 服务地址，默认 ``http://localhost:8000``
        timeout: HTTP 请求超时时间（秒），编排任务可能耗时较长
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def send_message(
        self,
        message_text: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """向主控 Agent 发送单条消息。

        Args:
            message_text: 用户消息内容
            task_id: 已有任务 ID，用于继续对话或恢复中断

        Returns:
            包含 status、detail、data、task_id 的结果字典
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._send_raw_message(client, message_text, task_id)

    async def check_connectivity(self) -> bool:
        """检查主控 Agent 是否可访问。"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/.well-known/agent-card.json")
                return resp.status_code == 200
        except Exception:
            return False

    async def _send_raw_message(
        self,
        client: httpx.AsyncClient,
        message_text: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """底层：发送 JSON-RPC SendMessage 请求。"""
        req = SendMessageRequest()
        if task_id:
            req.message.task_id = task_id
        req.message.message_id = uuid4().hex
        req.message.role = Role.ROLE_USER
        req.message.parts.add().text = message_text
        req.configuration.CopyFrom(SendMessageConfiguration())

        payload = {
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": json_format.MessageToDict(req),
            "id": 1,
        }
        resp = await client.post(
            self.base_url + "/",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "A2A-Version": "1.0",
            },
        )
        resp.raise_for_status()
        rpc_body = resp.json()

        if rpc_body.get("error"):
            return {
                "status": "failed",
                "detail": (
                    f"JSON-RPC Error {rpc_body['error'].get('code')}: "
                    f"{rpc_body['error'].get('message')}"
                ),
                "data": rpc_body,
                "task_id": task_id or "",
            }

        rpc_result = rpc_body.get("result", {})
        task = _extract_task_from_result(rpc_result)
        task_state = task.get("status", {}).get("state", "")
        state_name = _normalize_task_state(task_state)
        returned_task_id = task.get("id", task_id or "")

        return {
            "status": "passed",
            "detail": f"Task 状态: {state_name}",
            "data": task,
            "task_id": returned_task_id,
        }


def _extract_task_from_result(rpc_result: Dict[str, Any]) -> Dict[str, Any]:
    """从 JSON-RPC result 中提取 Task 对象（兼容 task 嵌套与扁平两种格式）。"""
    return rpc_result.get("task", rpc_result)


def _normalize_task_state(task_state: Any) -> str:
    """将 A2A Task 状态规范化为简短名称。"""
    if not isinstance(task_state, str):
        return str(task_state)
    return task_state.replace("TASK_STATE_", "").lower().replace("_", "-")


def _extract_parts_text(parts: List[Dict[str, Any]]) -> str:
    """从 message parts 中提取文本与文件链接。"""
    lines: List[str] = []
    for part in parts:
        text = part.get("text", "")
        if text:
            lines.append(text)
            continue
        url = part.get("url", "")
        if url:
            name = part.get("filename") or part.get("name") or "文件"
            lines.append(f"[{name}]({url})")
    return "\n\n".join(lines)


def _extract_invocation_traces(artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 artifacts 中提取调用轨迹。"""
    traces: List[Dict[str, Any]] = []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            text = part.get("text", "")
            if not text.startswith("__INVOCATION_TRACE__\n"):
                continue
            payload = text[len("__INVOCATION_TRACE__\n") :]
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                traces.extend(parsed)
    return traces


def parse_chat_response(result: Dict[str, Any]) -> ChatResponse:
    """将客户端原始响应解析为聊天展示结构。"""
    if result.get("status") != "passed":
        return ChatResponse(
            task_id=result.get("task_id", ""),
            state="failed",
            text=result.get("detail", "请求失败"),
            raw=result.get("data", {}),
            is_error=True,
        )

    data = result.get("data", {})
    task_id = result.get("task_id", data.get("id", ""))
    state = _normalize_task_state(data.get("status", {}).get("state", ""))

    status_message = data.get("status", {}).get("message", {})
    status_parts = status_message.get("parts", [])
    text = _extract_parts_text(status_parts)

    artifacts = data.get("artifacts", [])
    invocation_traces = _extract_invocation_traces(artifacts)
    artifact_texts: List[str] = []
    for artifact in artifacts:
        part_text = _extract_parts_text(artifact.get("parts", []))
        if part_text.startswith("__INVOCATION_TRACE__\n"):
            continue
        if part_text:
            artifact_texts.append(part_text)

    if state == "completed" and artifact_texts:
        # 总结通常在 status.message，artifacts 为各任务原始结果
        extra = "\n\n".join(artifact_texts[1:]) if len(artifact_texts) > 1 else ""
        if extra and extra not in text:
            text = f"{text}\n\n---\n\n{extra}" if text else extra
    elif not text and artifact_texts:
        text = "\n\n".join(artifact_texts)

    if state == "failed" and not text:
        text = "任务执行失败，请稍后重试。"

    if state == "input-required" and text:
        text = f"{text}\n\n请在下方输入补充信息后继续对话。"

    return ChatResponse(
        task_id=task_id,
        state=state,
        text=text or "（无响应内容）",
        artifacts=artifacts,
        invocation_traces=invocation_traces,
        raw=data,
        is_error=state == "failed",
    )
