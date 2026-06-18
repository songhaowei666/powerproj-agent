"""A2A 计划确认 Part 约定：text 展示 + data 驱动客户端按钮。"""

from typing import Any, Dict, List, Optional

from a2a_message_parser.confirmation import build_confirmation_data

PLAN_MEDIA_TYPE = "application/vnd.powerproj.plan+json"

DEFAULT_PLAN_CONFIRM_OPTIONS: List[Dict[str, str]] = [
    {"id": "approve", "label": "开始执行", "replyText": "确认执行"},
    {"id": "modify", "label": "修改计划", "replyText": "修改计划："},
    {"id": "cancel", "label": "取消", "replyText": "取消"},
]


def build_plan_confirm_data(
    body: Dict[str, Any],
    *,
    title: str = "执行计划确认",
    options: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """构建 plan_confirm 类型的 data 字段。"""
    payload = build_confirmation_data(
        "plan_confirm",
        title=title,
        options=options or DEFAULT_PLAN_CONFIRM_OPTIONS,
        body=body,
    )
    payload["type"] = "plan_confirm"
    return payload


def build_plan_confirm_parts(
    text: str,
    body: Dict[str, Any],
    *,
    title: str = "执行计划确认",
    options: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """构建 plan_confirm 的 input-required 消息 parts。"""
    parts: List[Dict[str, Any]] = [{"text": text}]
    parts.append(
        {
            "mediaType": PLAN_MEDIA_TYPE,
            "data": build_plan_confirm_data(
                body,
                title=title,
                options=options,
            ),
        }
    )
    return parts


def parse_plan_confirm_from_parts(
    parts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从 message parts 中解析 plan_confirm data。"""
    for part in parts:
        data = part.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("type") == "plan_confirm":
            return data
        media_type = part.get("mediaType") or part.get("media_type") or ""
        if PLAN_MEDIA_TYPE in media_type:
            return data
    return None
