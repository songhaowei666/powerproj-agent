"""A2A 服务验证器 — Streamlit 界面入口。

运行方式::

    streamlit run a2a_validator/app.py
"""

import asyncio

import streamlit as st

from a2a_validator.validator import A2AValidator

st.set_page_config(page_title="A2A 服务验证器", layout="wide")

st.title("🔍 A2A 服务验证器")
st.markdown("输入 A2A 服务地址，一键验证其协议合规性与功能可用性。")

base_url = st.text_input(
    "A2A 服务地址",
    value="http://localhost:8001",
    placeholder="http://localhost:8001",
)
test_message = st.text_input(
    "测试消息",
    value="你好，请简单回复",
)

# ---------- 完整验证 ----------
if st.button("开始验证", type="primary"):
    with st.spinner("正在验证，请稍候..."):
        validator = A2AValidator(base_url=base_url)
        results = asyncio.run(validator.validate(test_message=test_message))

    st.subheader("验证结果")

    for key, result in results.items():
        if key == "base_url":
            continue

        status = result.get("status", "unknown")
        detail = result.get("detail", "")

        if status == "passed":
            st.success(f"✅ **{key}**: {detail}")

            data = result.get("data")
            if key == "agent_card" and data is not None:
                with st.expander("查看 Agent Card 详情"):
                    card = data
                    st.json(
                        {
                            "name": card.name,
                            "description": card.description,
                            "version": card.version,
                            "capabilities": {
                                "streaming": (
                                    card.capabilities.streaming
                                    if card.capabilities
                                    else False
                                ),
                            },
                            "skills": [
                                {
                                    "id": s.id,
                                    "name": s.name,
                                    "description": s.description,
                                }
                                for s in card.skills
                            ],
                        }
                    )
            elif key == "streaming" and data is not None:
                with st.expander("查看流式响应详情"):
                    st.text_area(
                        "文本片段",
                        value="".join(data.get("chunks", [])),
                        height=150,
                    )
                    st.json({"artifacts_count": len(data.get("artifacts", []))})
            elif key == "single_message" and data is not None:
                with st.expander("查看单消息响应详情"):
                    st.json(data)
                # 显示 task_id，方便用户复制用于继续对话
                task_id = result.get("task_id", "")
                if task_id:
                    st.info(f"📋 **Task ID**: `{task_id}` （可复制用于继续对话）")

        elif status == "failed":
            st.error(f"❌ **{key}**: {detail}")
        elif status == "skipped":
            st.warning(f"⏭️ **{key}**: {detail}")
        else:
            st.info(f"⏳ **{key}**: {detail}")

st.divider()

# ---------- 单消息 / 继续对话 ----------
st.subheader("💬 单消息测试 / 继续对话")
st.markdown(
    "下方支持两种模式：\n"
    "1. **新建任务**：不填 Task ID，发送新消息创建任务\n"
    "2. **继续对话**：填入已有 Task ID，发送补充消息继续对话（用于 input-required 恢复）"
)

col1, col2 = st.columns([1, 2])
with col1:
    manual_task_id = st.text_input(
        "Task ID（可选，用于继续已有对话）",
        placeholder="留空则创建新任务",
        help="当服务端返回 input-required 时，复制上一次的 Task ID 填入此处，然后输入补充消息",
    )
with col2:
    followup_message = st.text_input(
        "消息内容",
        placeholder="输入消息内容...",
    )

if st.button("发送消息", type="secondary"):
    if not followup_message.strip():
        st.warning("请输入消息内容")
    else:
        with st.spinner("正在发送..."):
            validator = A2AValidator(base_url=base_url)
            task_id = manual_task_id.strip() or None
            result = asyncio.run(
                validator.send_message(
                    message_text=followup_message.strip(),
                    task_id=task_id,
                )
            )

        status = result.get("status", "unknown")
        detail = result.get("detail", "")
        resp_data = result.get("data", {})
        returned_task_id = result.get("task_id", "")

        if status == "passed":
            st.success(f"✅ **成功**: {detail}")
            if returned_task_id:
                st.info(f"📋 **Task ID**: `{returned_task_id}`")
            with st.expander("查看响应详情"):
                st.json(resp_data)
        else:
            st.error(f"❌ **失败**: {detail}")
            with st.expander("查看错误详情"):
                st.json(resp_data)
