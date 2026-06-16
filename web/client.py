"""主控 Agent A2A 聊天客户端（流式）。"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import google.protobuf.json_format as json_format
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_message_text
from a2a.types import Message, Part, Role, SendMessageConfiguration, SendMessageRequest

from main_agent.streaming import (
    parse_summary_chunk_message,
    parse_trace_step_message,
)
from a2a_message_parser import parse_confirmation_from_parts


@dataclass
class StreamEvent:
    """流式事件。"""

    event_type: str
    text: str = ""
    trace: Optional[Dict[str, Any]] = None
    summary_chunk: str = ""
    artifact: Optional[Dict[str, Any]] = None
    task_state: str = ""


@dataclass
class ConfirmationUI:
    """结构化确认交互（是/否等按钮）。"""

    action: str
    options: List[Dict[str, str]]
    title: str = ""


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
    confirmation: Optional[ConfirmationUI] = None


class MainAgentClient:
    """通过 A2A 流式 JSON-RPC 与主控 Agent 通信的客户端。

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
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> Dict[str, Any]:
        """向主控 Agent 发送单条消息（流式接收）。

        Args:
            message_text: 用户消息内容
            task_id: 已有任务 ID，用于继续对话或恢复中断
            on_event: 流式事件回调，用于 UI 实时刷新

        Returns:
            包含 status、detail、data、task_id 的结果字典
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._send_streaming_message(client, message_text, task_id, on_event)

    async def check_connectivity(self) -> bool:
        """检查主控 Agent 是否可访问。"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/.well-known/agent-card.json")
                return resp.status_code == 200
        except Exception:
            return False

    async def _send_streaming_message(
        self,
        client: httpx.AsyncClient,
        message_text: str,
        task_id: Optional[str] = None,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> Dict[str, Any]:
        """底层：通过 A2A SDK 流式发送 SendMessage 请求。"""
        try:
            resolver = A2ACardResolver(httpx_client=client, base_url=self.base_url)
            agent_card = await resolver.get_agent_card()
        except Exception as exc:
            return {
                "status": "failed",
                "detail": f"无法获取 Agent Card：{type(exc).__name__}: {exc}",
                "data": {},
                "task_id": task_id or "",
            }

        config = ClientConfig(httpx_client=client, streaming=True)
        a2a_client = ClientFactory(config).create(agent_card)

        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=message_text)],
            message_id=uuid4().hex,
        )
        if task_id:
            message.task_id = task_id

        request = SendMessageRequest(
            message=message,
            configuration=SendMessageConfiguration(),
        )

        invocation_traces: List[Dict[str, Any]] = []
        summary_parts: List[str] = []
        artifacts: List[Dict[str, Any]] = []
        task_data: Dict[str, Any] = {}
        returned_task_id = task_id or ""
        task_state = ""

        try:
            async for response in a2a_client.send_message(request):
                if response.HasField("status_update"):
                    event = response.status_update
                    returned_task_id = event.task_id or returned_task_id
                    status = event.status
                    task_state = _normalize_task_state(status.state)

                    status_message = status.message if status.HasField("message") else None
                    chunk_text = get_message_text(status_message) if status_message else ""

                    if chunk_text.startswith("正在处理"):
                        stream_event = StreamEvent(
                            event_type="working",
                            text=chunk_text,
                            task_state=task_state,
                        )
                        if on_event:
                            on_event(stream_event)
                        continue

                    trace_dict = parse_trace_step_message(chunk_text)
                    if trace_dict is not None:
                        invocation_traces.append(trace_dict)
                        stream_event = StreamEvent(
                            event_type="trace",
                            trace=trace_dict,
                            task_state=task_state,
                        )
                        if on_event:
                            on_event(stream_event)
                        continue

                    summary_chunk = parse_summary_chunk_message(chunk_text)
                    if summary_chunk is not None:
                        summary_parts.append(summary_chunk)
                        stream_event = StreamEvent(
                            event_type="summary",
                            summary_chunk=summary_chunk,
                            task_state=task_state,
                        )
                        if on_event:
                            on_event(stream_event)
                        continue

                    if chunk_text:
                        stream_event = StreamEvent(
                            event_type="status",
                            text=chunk_text,
                            task_state=task_state,
                        )
                        if on_event:
                            on_event(stream_event)

                    task_data = {
                        "id": returned_task_id,
                        "status": json_format.MessageToDict(status),
                        "artifacts": artifacts,
                    }

                elif response.HasField("artifact_update"):
                    event = response.artifact_update
                    returned_task_id = event.task_id or returned_task_id
                    artifact_dict = json_format.MessageToDict(event.artifact)
                    artifacts.append(artifact_dict)
                    stream_event = StreamEvent(
                        event_type="artifact",
                        artifact=artifact_dict,
                        task_state=task_state,
                    )
                    if on_event:
                        on_event(stream_event)
                    task_data = {
                        "id": returned_task_id,
                        "status": task_data.get("status", {}),
                        "artifacts": artifacts,
                    }

        except Exception as exc:
            return {
                "status": "failed",
                "detail": f"流式请求失败：{type(exc).__name__}: {exc}",
                "data": task_data,
                "task_id": returned_task_id,
                "invocation_traces": invocation_traces,
                "summary_parts": summary_parts,
            }

        if not task_data:
            task_data = {
                "id": returned_task_id,
                "status": {"state": task_state},
                "artifacts": artifacts,
            }

        return {
            "status": "passed",
            "detail": f"Task 状态: {task_state or 'unknown'}",
            "data": task_data,
            "task_id": returned_task_id,
            "invocation_traces": invocation_traces,
            "summary_parts": summary_parts,
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


def _extract_invocation_traces_from_artifacts(
    artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从 artifacts 中提取调用轨迹（失败兜底）。"""
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


def _build_confirmation_ui(
    parts: List[Dict[str, Any]],
) -> Optional[ConfirmationUI]:
    """从 parts 中解析 confirmation 交互定义。"""
    data = parse_confirmation_from_parts(parts)
    if not data:
        return None
    options = data.get("options") or []
    if not options:
        return None
    normalized_options: List[Dict[str, str]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label", "")).strip()
        reply_text = str(option.get("replyText", label)).strip()
        if label:
            normalized_options.append(
                {
                    "id": str(option.get("id", label)),
                    "label": label,
                    "replyText": reply_text or label,
                }
            )
    if not normalized_options:
        return None
    return ConfirmationUI(
        action=str(data.get("action", "")),
        title=str(data.get("title", "")),
        options=normalized_options,
    )


def parse_chat_response(result: Dict[str, Any]) -> ChatResponse:
    """将客户端原始响应解析为聊天展示结构。"""
    if result.get("status") != "passed":
        return ChatResponse(
            task_id=result.get("task_id", ""),
            state="failed",
            text=result.get("detail", "请求失败"),
            invocation_traces=result.get("invocation_traces", []),
            raw=result.get("data", {}),
            is_error=True,
        )

    data = result.get("data", {})
    task_id = result.get("task_id", data.get("id", ""))
    state = _normalize_task_state(data.get("status", {}).get("state", ""))

    status_message = data.get("status", {}).get("message", {})
    status_parts = status_message.get("parts", [])
    text = _extract_parts_text(status_parts)
    confirmation = _build_confirmation_ui(status_parts)

    artifacts = data.get("artifacts", [])
    streamed_traces = result.get("invocation_traces", [])
    artifact_traces = _extract_invocation_traces_from_artifacts(artifacts)
    invocation_traces = streamed_traces or artifact_traces

    summary_parts = result.get("summary_parts", [])
    streamed_summary = "".join(summary_parts).strip()
    if streamed_summary:
        text = streamed_summary
    elif state == "input-required" and text and not confirmation:
        text = f"{text}\n\n请在下方输入补充信息后继续对话。"

    artifact_texts: List[str] = []
    for artifact in artifacts:
        part_text = _extract_parts_text(artifact.get("parts", []))
        if part_text.startswith("__INVOCATION_TRACE__\n"):
            continue
        if part_text:
            artifact_texts.append(part_text)

    if state == "completed" and artifact_texts and not streamed_summary:
        extra = "\n\n".join(artifact_texts[1:]) if len(artifact_texts) > 1 else ""
        if extra and extra not in text:
            text = f"{text}\n\n---\n\n{extra}" if text else extra
    elif not text and artifact_texts:
        text = "\n\n".join(artifact_texts)

    if state == "failed" and not text:
        text = "任务执行失败，请稍后重试。"

    if state == "input-required" and text and not confirmation and "请在下方输入补充信息" not in text:
        text = f"{text}\n\n请在下方输入补充信息后继续对话。"

    return ChatResponse(
        task_id=task_id,
        state=state,
        text=text or "（无响应内容）",
        artifacts=artifacts,
        invocation_traces=invocation_traces,
        raw=data,
        is_error=state == "failed",
        confirmation=confirmation if state == "input-required" else None,
    )
