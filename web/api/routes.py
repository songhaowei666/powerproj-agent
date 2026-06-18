"""Web API 路由：连通性检查与 SSE 聊天。"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from web.api.schemas import ChatRequest, ConnectivityResponse
from web.client import MainAgentClient, StreamEvent, chat_response_to_dict, parse_chat_response

router = APIRouter(prefix="/api")

DEFAULT_BASE_URL = "http://localhost:8000"


def _format_sse(event: str, data: dict) -> str:
    """格式化 SSE 事件行。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/connectivity", response_model=ConnectivityResponse)
async def check_connectivity(base_url: str = DEFAULT_BASE_URL) -> ConnectivityResponse:
    """检查主控 Agent 是否在线。"""
    normalized_url = base_url.rstrip("/")
    client = MainAgentClient(base_url=normalized_url)
    online = await client.check_connectivity()
    return ConnectivityResponse(online=online, base_url=normalized_url)


async def _stream_chat_events(request: ChatRequest) -> AsyncGenerator[str, None]:
    """流式发送聊天消息并 yield SSE 事件。"""
    base_url = (request.base_url or DEFAULT_BASE_URL).rstrip("/")
    client = MainAgentClient(base_url=base_url)
    event_queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    def _on_event(event: StreamEvent) -> None:
        if event.event_type == "trace" and event.trace:
            event_queue.put_nowait(("trace", event.trace))
        elif event.event_type == "summary" and event.summary_chunk:
            event_queue.put_nowait(("summary", {"chunk": event.summary_chunk}))

    async def _run_send() -> None:
        try:
            raw_result = await client.send_message(
                request.message,
                task_id=request.task_id,
                context_id=request.context_id,
                on_event=_on_event,
            )
            chat_resp = parse_chat_response(raw_result)
            await event_queue.put(("done", chat_response_to_dict(chat_resp)))
        except Exception as exc:
            await event_queue.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            await event_queue.put(None)

    send_task = asyncio.create_task(_run_send())

    try:
        while True:
            item = await event_queue.get()
            if item is None:
                break
            event_name, payload = item
            yield _format_sse(event_name, payload)
    finally:
        if not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass


@router.post("/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口（SSE）。"""
    return StreamingResponse(
        _stream_chat_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
