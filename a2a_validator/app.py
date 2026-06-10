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
    value="http://localhost:9999",
    placeholder="http://localhost:9999",
)
test_message = st.text_input(
    "测试消息",
    value="你好，请简单回复",
)

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
                    st.json({"responses_count": len(data)})
                    for idx, resp in enumerate(data):
                        with st.container(border=True):
                            st.caption(f"响应 #{idx + 1}")
                            if hasattr(resp, "message") and resp.message:
                                from a2a.helpers import get_message_text
                                text = get_message_text(resp.message)
                                st.markdown(f"**消息文本**: {text}")
                            if hasattr(resp, "status") and resp.status:
                                st.json({
                                    "state": resp.status.state,
                                    "message_text": (
                                        get_message_text(resp.status.message)
                                        if resp.status.HasField("message")
                                        else None
                                    ),
                                })
                            if hasattr(resp, "artifacts") and resp.artifacts:
                                st.markdown(f"**Artifacts 数量**: {len(resp.artifacts)}")
                                for art_idx, art in enumerate(resp.artifacts):
                                    art_parts = []
                                    for part in art.parts:
                                        if part.HasField("text"):
                                            art_parts.append({"type": "text", "content": part.text})
                                        elif part.HasField("url"):
                                            art_parts.append({"type": "file", "url": part.url, "filename": part.filename})
                                    st.json({f"artifact_{art_idx}": art_parts})

        elif status == "failed":
            st.error(f"❌ **{key}**: {detail}")
        elif status == "skipped":
            st.warning(f"⏭️ **{key}**: {detail}")
        else:
            st.info(f"⏳ **{key}**: {detail}")
