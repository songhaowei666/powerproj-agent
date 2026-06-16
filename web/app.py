"""主控 Agent 聊天页面 — Streamlit 界面入口。

运行方式::

    streamlit run web/app.py

使用前请先启动主控 Agent 及下游业务 Agent::

    python main_agent/server.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from web.client import MainAgentClient, StreamEvent, parse_chat_response, ConfirmationUI

DEFAULT_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="电网智能助手",
    page_icon=None,
    layout="centered",
)

st.title("电网智能助手")
st.caption("通过主控 Agent 统一调度统计、规划、投资等业务能力")


def _init_session_state() -> None:
    """初始化会话状态。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "task_id" not in st.session_state:
        st.session_state.task_id = None
    if "awaiting_input" not in st.session_state:
        st.session_state.awaiting_input = False
    if "pending_confirmation" not in st.session_state:
        st.session_state.pending_confirmation = None
    if "confirmation_reply" not in st.session_state:
        st.session_state.confirmation_reply = None
    if "base_url" not in st.session_state:
        st.session_state.base_url = DEFAULT_BASE_URL


def _reset_conversation() -> None:
    """清空当前对话。"""
    st.session_state.messages = []
    st.session_state.task_id = None
    st.session_state.awaiting_input = False
    st.session_state.pending_confirmation = None
    st.session_state.confirmation_reply = None


def _render_sidebar() -> str:
    """渲染侧边栏并返回当前服务地址。"""
    with st.sidebar:
        st.header("设置")
        base_url = st.text_input(
            "主控 Agent 地址",
            value=st.session_state.base_url,
            placeholder=DEFAULT_BASE_URL,
        )
        st.session_state.base_url = base_url.rstrip("/")

        if st.button("新对话", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()
        st.markdown("**使用说明**")
        st.markdown(
            "1. 先启动主控 Agent（端口 8000）及业务 Agent\n"
            "2. 输入自然语言问题，例如统计、规划、投资相关需求\n"
            "3. 若 Agent 需要补充信息，直接在对话框继续回复即可\n"
            "4. 调用轨迹与总结会在处理过程中实时更新"
        )

        if st.session_state.task_id:
            st.divider()
            st.text("当前任务 ID")
            st.code(st.session_state.task_id, language=None)

    return st.session_state.base_url


def _render_invocation_traces(traces: list[dict]) -> None:
    """渲染意图识别与业务 Agent 的调用轨迹。"""
    if not traces:
        return

    st.markdown("#### 调用轨迹")
    sorted_traces = sorted(traces, key=lambda item: item.get("step", 0))
    for trace in sorted_traces:
        agent_type = trace.get("agent_type", "")
        agent_name = trace.get("agent_name", "未知 Agent")
        step = trace.get("step", 0)
        status = trace.get("status", "success")
        status_label = "成功" if status == "success" else "失败"

        if agent_type == "intent":
            title = f"步骤 {step} | 意图识别 Agent | {status_label}"
        else:
            capability = trace.get("capability", "")
            task_id = trace.get("task_id", "")
            phase = trace.get("phase")
            phase_text = f"Phase {phase}" if phase is not None else ""
            title = (
                f"步骤 {step} | 业务 Agent: {agent_name}"
                f" | 能力: {capability} | 任务: {task_id}"
                f" | {phase_text} | {status_label}"
            )

        with st.expander(title, expanded=False):
            col_input, col_output = st.columns(2)
            with col_input:
                st.markdown("**调用参数**")
                st.json(trace.get("input", {}))
            with col_output:
                st.markdown("**返回结果**")
                st.json(trace.get("output", {}))


def _check_service_online(base_url: str) -> bool:
    """检查主控 Agent 是否在线。"""
    client = MainAgentClient(base_url=base_url)
    return asyncio.run(client.check_connectivity())


async def _send_streaming_to_main_agent(
    base_url: str,
    message: str,
    task_id: str | None,
    on_event,
):
    """流式发送消息到主控 Agent。"""
    client = MainAgentClient(base_url=base_url)
    return await client.send_message(message, task_id, on_event=on_event)


def _render_confirmation_buttons(confirmation: ConfirmationUI, task_id: str) -> None:
    """渲染是/否等确认按钮，点击后写入 confirmation_reply 触发发送。"""
    if confirmation.title:
        st.markdown(f"**{confirmation.title}**")
    cols = st.columns(len(confirmation.options))
    for index, option in enumerate(confirmation.options):
        with cols[index]:
            if st.button(
                option["label"],
                key=f"confirm_{task_id}_{option['id']}_{index}",
                use_container_width=True,
            ):
                st.session_state.confirmation_reply = option["replyText"]
                st.rerun()


def _process_assistant_response(
    base_url: str,
    trace_slot,
    text_slot,
    *,
    user_prompt: str,
    send_task_id: str | None,
) -> None:
    """发送消息并渲染助手回复（含确认按钮）。"""
    try:
        raw_result = _send_to_main_agent(
            base_url, user_prompt, send_task_id, trace_slot, text_slot
        )
        chat_resp = parse_chat_response(raw_result)
        st.session_state.awaiting_input = chat_resp.state == "input-required"
        if st.session_state.awaiting_input:
            st.session_state.task_id = chat_resp.task_id or None
            st.session_state.pending_confirmation = chat_resp.confirmation
        else:
            st.session_state.task_id = None
            st.session_state.pending_confirmation = None

        if chat_resp.is_error:
            trace_slot.empty()
            text_slot.error(chat_resp.text)
        else:
            with trace_slot.container():
                _render_invocation_traces(chat_resp.invocation_traces)
            text_slot.markdown(chat_resp.text)
            if chat_resp.confirmation and chat_resp.task_id:
                _render_confirmation_buttons(chat_resp.confirmation, chat_resp.task_id)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": chat_resp.text,
                "invocation_traces": chat_resp.invocation_traces,
                "confirmation": chat_resp.confirmation,
                "task_id": chat_resp.task_id,
            }
        )
    except Exception as exc:
        error_text = f"请求失败：{type(exc).__name__}: {exc}"
        text_slot.error(error_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_text}
        )


def _send_to_main_agent(
    base_url: str,
    message: str,
    task_id: str | None,
    trace_slot,
    text_slot,
):
    """同步包装：流式请求并在占位符中实时刷新（轨迹在上，正文在下）。"""
    traces: list[dict] = []
    summary_parts: list[str] = []

    def _on_event(event: StreamEvent) -> None:
        if event.event_type == "trace" and event.trace:
            traces.append(event.trace)
            with trace_slot.container():
                _render_invocation_traces(traces)
        elif event.event_type == "summary" and event.summary_chunk:
            summary_parts.append(event.summary_chunk)
            with text_slot.container():
                st.markdown("".join(summary_parts))

    return asyncio.run(
        _send_streaming_to_main_agent(base_url, message, task_id, _on_event)
    )


_init_session_state()
base_url = _render_sidebar()

is_online = _check_service_online(base_url)
if is_online:
    st.success(f"已连接主控 Agent：{base_url}")
else:
    st.warning(
        f"无法连接主控 Agent（{base_url}），请确认服务已启动："
        "`python main_agent/server.py`"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("invocation_traces"):
            _render_invocation_traces(message["invocation_traces"])
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and st.session_state.awaiting_input
            and message.get("confirmation")
            and message.get("task_id") == st.session_state.task_id
        ):
            _render_confirmation_buttons(message["confirmation"], message["task_id"])

if st.session_state.confirmation_reply:
    reply_text = st.session_state.confirmation_reply
    st.session_state.confirmation_reply = None
    st.session_state.messages.append({"role": "user", "content": reply_text})
    with st.chat_message("user"):
        st.markdown(reply_text)
    with st.chat_message("assistant"):
        if not is_online:
            error_text = "主控 Agent 未连接，请先启动服务后再试。"
            st.error(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text}
            )
        else:
            trace_slot = st.empty()
            text_slot = st.empty()
            text_slot.markdown("_正在处理，请稍候..._")
            _process_assistant_response(
                base_url,
                trace_slot,
                text_slot,
                user_prompt=reply_text,
                send_task_id=st.session_state.task_id,
            )

elif prompt := st.chat_input(
    "请输入您的问题，例如：帮我统计今年的投资收益并做明年规划",
    disabled=bool(st.session_state.pending_confirmation),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not is_online:
            error_text = "主控 Agent 未连接，请先启动服务后再试。"
            st.error(error_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text}
            )
        else:
            trace_slot = st.empty()
            text_slot = st.empty()
            text_slot.markdown("_正在处理，请稍候..._")

            send_task_id = (
                st.session_state.task_id
                if st.session_state.awaiting_input
                else None
            )
            _process_assistant_response(
                base_url,
                trace_slot,
                text_slot,
                user_prompt=prompt,
                send_task_id=send_task_id,
            )
