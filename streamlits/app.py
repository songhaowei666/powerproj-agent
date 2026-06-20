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

from streamlits.client import MainAgentClient, StreamEvent, parse_chat_response, ConfirmationUI
from main_agent.task_manager import format_plan_approve_reply

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
    if "context_id" not in st.session_state:
        st.session_state.context_id = None
    if "awaiting_input" not in st.session_state:
        st.session_state.awaiting_input = False
    if "pending_confirmation" not in st.session_state:
        st.session_state.pending_confirmation = None
    if "confirmation_reply" not in st.session_state:
        st.session_state.confirmation_reply = None
    if "seen_trace_steps" not in st.session_state:
        st.session_state.seen_trace_steps = set()
    if "base_url" not in st.session_state:
        st.session_state.base_url = DEFAULT_BASE_URL


def _reset_conversation() -> None:
    """清空当前对话。"""
    st.session_state.messages = []
    st.session_state.task_id = None
    st.session_state.context_id = None
    st.session_state.awaiting_input = False
    st.session_state.pending_confirmation = None
    st.session_state.confirmation_reply = None
    st.session_state.seen_trace_steps = set()


def _reset_trace_tracking() -> None:
    """新任务开始时清空已展示的轨迹步骤号。"""
    st.session_state.seen_trace_steps = set()


def _extract_new_traces(traces: list[dict]) -> list[dict]:
    """从流式/最终轨迹中筛出本轮尚未展示的新步骤。"""
    if not traces:
        return []
    seen: set[int] = st.session_state.seen_trace_steps
    new_traces: list[dict] = []
    for trace in sorted(traces, key=lambda item: item.get("step", 0)):
        step = trace.get("step")
        if isinstance(step, int) and step > 0:
            if step in seen:
                continue
            seen.add(step)
        new_traces.append(trace)
    st.session_state.seen_trace_steps = seen
    return new_traces


def _format_trace_status_label(status: str) -> str:
    """将轨迹状态转为中文标签。"""
    mapping = {
        "success": "成功",
        "failed": "失败",
        "input_required": "待确认",
    }
    return mapping.get(status, status or "未知")


def _build_trace_title(trace: dict, *, turn_index: int, intent_index: int) -> str:
    """构建单条轨迹折叠标题。"""
    agent_type = trace.get("agent_type", "")
    agent_name = trace.get("agent_name", "未知 Agent")
    global_step = trace.get("step", turn_index)
    status_label = _format_trace_status_label(trace.get("status", "success"))

    if agent_type == "intent":
        if intent_index > 1:
            role_text = "意图识别（补充后）"
        else:
            role_text = "意图识别"
        return f"本轮 {turn_index} | {role_text} | {status_label}（总步骤 {global_step}）"

    capability = trace.get("capability", "")
    task_id = trace.get("task_id", "")
    phase = trace.get("phase")
    phase_text = f"Phase {phase}" if phase is not None else ""
    return (
        f"本轮 {turn_index} | 业务 Agent: {agent_name}"
        f" | 能力: {capability} | 任务: {task_id}"
        f" | {phase_text} | {status_label}（总步骤 {global_step}）"
    )


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
            "3. 计划确认时可勾选子任务并点「开始执行」，业务确认可点「是/否」\n"
            "4. 调用轨迹与总结会在处理过程中实时更新"
        )

        if st.session_state.task_id:
            st.divider()
            st.text("当前任务 ID")
            st.code(st.session_state.task_id, language=None)

    return st.session_state.base_url


def _render_invocation_traces(traces: list[dict]) -> None:
    """渲染本轮新增的调用轨迹。"""
    if not traces:
        return

    st.markdown("#### 本轮调用轨迹")
    sorted_traces = sorted(traces, key=lambda item: item.get("step", 0))
    intent_seen = 0
    for turn_index, trace in enumerate(sorted_traces, start=1):
        if trace.get("agent_type") == "intent":
            intent_seen += 1
        title = _build_trace_title(trace, turn_index=turn_index, intent_index=intent_seen)

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
    context_id: str | None,
    on_event,
):
    """流式发送消息到主控 Agent。"""
    client = MainAgentClient(base_url=base_url)
    return await client.send_message(
        message,
        task_id,
        context_id=context_id,
        on_event=on_event,
    )


def _is_awaiting_confirmation() -> bool:
    """判断是否处于待确认（是/否按钮）状态。"""
    if st.session_state.pending_confirmation:
        return True
    if not st.session_state.awaiting_input or not st.session_state.task_id:
        return False
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant":
            return bool(
                message.get("confirmation")
                and message.get("task_id") == st.session_state.task_id
            )
        if message.get("role") == "user":
            return False
    return False


def _resolve_chat_send_task_id() -> str | None:
    """解析输入框发送时应使用的 task_id。

    - 待确认时输入新问题：放弃确认，开启新任务
    - 普通 input-required（文本补全）：续传 task_id
    - 其他：新任务
    """
    if _is_awaiting_confirmation():
        st.session_state.task_id = None
        st.session_state.context_id = None
        st.session_state.awaiting_input = False
        st.session_state.pending_confirmation = None
        return None
    if st.session_state.awaiting_input:
        return st.session_state.task_id
    return None


def _resolve_chat_send_context_id() -> str | None:
    """解析续传时应携带的 context_id。"""
    if st.session_state.awaiting_input and st.session_state.task_id:
        return st.session_state.context_id
    return None


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
                st.session_state.pending_confirmation = None
                st.rerun()


def _render_plan_confirm_ui(confirmation: ConfirmationUI, task_id: str) -> None:
    """渲染计划确认勾选框、修改说明输入区与操作按钮。"""
    body = confirmation.body or {}
    tasks = body.get("tasks") or []
    revision = int(body.get("revision") or 1)
    task_items = [
        task for task in tasks if isinstance(task, dict) and task.get("id")
    ]

    if confirmation.title:
        st.markdown(f"**{confirmation.title}**")

    checkbox_items: list[tuple[str, str]] = []
    for index, task in enumerate(task_items):
        task_id_value = str(task["id"])
        checkbox_key = f"plan_task_{task_id}_{revision}_{task_id_value}_{index}"
        agent_name = str(task.get("required_agent", ""))
        task_name = str(task.get("name", ""))
        label = f"[{agent_name}] {task_name}" if agent_name else task_name
        st.checkbox(label, value=True, key=checkbox_key)
        checkbox_items.append((checkbox_key, task_id_value))

    modify_text_key = f"plan_modify_text_{task_id}_{revision}"
    st.text_area(
        "修改说明（如需调整计划，请描述后再点「提交修改」）",
        key=modify_text_key,
        placeholder="例如：去掉投资分析，增加规划 agent 查询项目信息",
        height=88,
    )

    all_task_ids = [item[1] for item in checkbox_items]
    approve_option = next(
        (option for option in confirmation.options if option.get("id") == "approve"),
        None,
    )
    cancel_option = next(
        (option for option in confirmation.options if option.get("id") == "cancel"),
        None,
    )

    cols = st.columns(3)
    col_index = 0

    if approve_option:
        selected_ids = [
            task_id_value
            for checkbox_key, task_id_value in checkbox_items
            if st.session_state.get(checkbox_key, True)
        ]
        with cols[col_index]:
            if st.button(
                approve_option["label"],
                key=f"plan_approve_{task_id}_{revision}",
                use_container_width=True,
                disabled=not selected_ids,
            ):
                st.session_state.confirmation_reply = format_plan_approve_reply(
                    selected_ids,
                    all_task_ids,
                )
                st.session_state.pending_confirmation = None
                st.rerun()
        col_index += 1

    with cols[col_index]:
        if st.button(
            "提交修改",
            key=f"plan_submit_modify_{task_id}_{revision}",
            use_container_width=True,
        ):
            modify_text = str(st.session_state.get(modify_text_key, "") or "").strip()
            if not modify_text:
                st.warning("请填写修改说明后再提交")
            else:
                st.session_state.confirmation_reply = f"修改计划：{modify_text}"
                st.session_state.pending_confirmation = None
                st.rerun()
    col_index += 1

    if cancel_option:
        with cols[col_index]:
            if st.button(
                cancel_option["label"],
                key=f"plan_action_{task_id}_{revision}_cancel",
                use_container_width=True,
            ):
                st.session_state.confirmation_reply = cancel_option["replyText"]
                st.session_state.pending_confirmation = None
                st.rerun()


def _render_confirmation_ui(confirmation: ConfirmationUI, task_id: str) -> None:
    """根据确认类型渲染按钮或计划勾选框。"""
    if confirmation.confirm_type == "plan_confirm":
        _render_plan_confirm_ui(confirmation, task_id)
        return
    _render_confirmation_buttons(confirmation, task_id)


def _format_assistant_message_content(content: str, confirmation: ConfirmationUI | None) -> str:
    """计划确认时仅展示引导语，子任务清单由勾选框承担。"""
    if confirmation and confirmation.confirm_type == "plan_confirm" and content:
        return content.split("\n\n", 1)[0]
    return content


def _process_assistant_response(
    base_url: str,
    trace_slot,
    text_slot,
    *,
    user_prompt: str,
    send_task_id: str | None,
    send_context_id: str | None,
) -> None:
    """发送消息并渲染助手回复（含确认按钮）。"""
    if send_task_id is None:
        _reset_trace_tracking()

    turn_traces: list[dict] = []

    try:
        raw_result = _send_to_main_agent(
            base_url,
            user_prompt,
            send_task_id,
            send_context_id,
            trace_slot,
            text_slot,
            turn_traces,
        )
        chat_resp = parse_chat_response(raw_result)
        if not turn_traces and chat_resp.invocation_traces:
            turn_traces = _extract_new_traces(chat_resp.invocation_traces)
        st.session_state.awaiting_input = chat_resp.state == "input-required"
        if st.session_state.awaiting_input:
            st.session_state.task_id = chat_resp.task_id or None
            st.session_state.context_id = chat_resp.context_id or None
            st.session_state.pending_confirmation = chat_resp.confirmation
        else:
            st.session_state.task_id = None
            st.session_state.context_id = None
            st.session_state.pending_confirmation = None

        if chat_resp.is_error:
            trace_slot.empty()
            text_slot.error(chat_resp.text)
        else:
            with trace_slot.container():
                _render_invocation_traces(turn_traces)
            text_slot.markdown(
                _format_assistant_message_content(chat_resp.text, chat_resp.confirmation)
            )
            if chat_resp.confirmation and chat_resp.task_id:
                _render_confirmation_ui(chat_resp.confirmation, chat_resp.task_id)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": chat_resp.text,
                "invocation_traces": turn_traces,
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
    context_id: str | None,
    trace_slot,
    text_slot,
    turn_traces: list[dict],
):
    """同步包装：流式请求并在占位符中实时刷新（轨迹在上，正文在下）。"""
    summary_parts: list[str] = []

    def _on_event(event: StreamEvent) -> None:
        if event.event_type == "trace" and event.trace:
            new_items = _extract_new_traces([event.trace])
            if new_items:
                turn_traces.extend(new_items)
                with trace_slot.container():
                    _render_invocation_traces(turn_traces)
        elif event.event_type == "summary" and event.summary_chunk:
            summary_parts.append(event.summary_chunk)
            with text_slot.container():
                st.markdown("".join(summary_parts))

    return asyncio.run(
        _send_streaming_to_main_agent(
            base_url,
            message,
            task_id,
            context_id,
            _on_event,
        )
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
        display_content = _format_assistant_message_content(
            message["content"],
            message.get("confirmation"),
        )
        st.markdown(display_content)
        if (
            message["role"] == "assistant"
            and st.session_state.awaiting_input
            and message.get("confirmation")
            and message.get("task_id") == st.session_state.task_id
        ):
            _render_confirmation_ui(message["confirmation"], message["task_id"])

if st.session_state.confirmation_reply:
    reply_text = st.session_state.confirmation_reply
    st.session_state.confirmation_reply = None
    st.session_state.pending_confirmation = None
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
                send_context_id=st.session_state.context_id,
            )

prompt = st.chat_input(
    "请输入您的问题，例如：帮我统计今年的投资收益并做明年规划",
)
if prompt:
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

            send_task_id = _resolve_chat_send_task_id()
            send_context_id = _resolve_chat_send_context_id()
            _process_assistant_response(
                base_url,
                trace_slot,
                text_slot,
                user_prompt=prompt,
                send_task_id=send_task_id,
                send_context_id=send_context_id,
            )
