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

from web.client import MainAgentClient, parse_chat_response

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
    if "base_url" not in st.session_state:
        st.session_state.base_url = DEFAULT_BASE_URL


def _reset_conversation() -> None:
    """清空当前对话。"""
    st.session_state.messages = []
    st.session_state.task_id = None


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
            "3. 若 Agent 需要补充信息，直接在对话框继续回复即可"
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
    for trace in traces:
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


def _send_to_main_agent(base_url: str, message: str, task_id: str | None):
    """发送消息到主控 Agent。"""
    client = MainAgentClient(base_url=base_url)
    return asyncio.run(client.send_message(message, task_id))


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
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("invocation_traces"):
            _render_invocation_traces(message["invocation_traces"])

if prompt := st.chat_input("请输入您的问题，例如：帮我统计今年的投资收益并做明年规划"):
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
            with st.spinner("正在处理，请稍候..."):
                try:
                    raw_result = _send_to_main_agent(
                        base_url, prompt, st.session_state.task_id
                    )
                    chat_resp = parse_chat_response(raw_result)
                    st.session_state.task_id = chat_resp.task_id or None

                    if chat_resp.is_error:
                        st.error(chat_resp.text)
                    else:
                        st.markdown(chat_resp.text)
                        _render_invocation_traces(chat_resp.invocation_traces)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": chat_resp.text,
                            "invocation_traces": chat_resp.invocation_traces,
                        }
                    )
                except Exception as exc:
                    error_text = f"请求失败：{type(exc).__name__}: {exc}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text}
                    )
