"""A2A 确认交互 Part 约定：text 展示 + data 驱动客户端按钮。"""

from typing import Any, Dict, List, Optional

CONFIRMATION_MEDIA_TYPE = "application/vnd.powerproj.confirmation+json"

DEFAULT_YES_NO_OPTIONS: List[Dict[str, str]] = [
    {"id": "yes", "label": "是", "replyText": "是"},
    {"id": "no", "label": "否", "replyText": "否"},
]

DELETE_CONFIRM_OPTIONS: List[Dict[str, str]] = [
    {"id": "confirm", "label": "确认删除", "replyText": "确认删除"},
    {"id": "cancel", "label": "取消", "replyText": "取消"},
]


def build_confirmation_data(
    action: str,
    *,
    title: str = "",
    options: Optional[List[Dict[str, str]]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建 confirmation 类型的 data 字段。"""
    payload: Dict[str, Any] = {
        "type": "confirmation",
        "action": action,
        "options": options or DEFAULT_YES_NO_OPTIONS,
    }
    if title:
        payload["title"] = title
    if body:
        payload["body"] = body
    return payload


def build_confirmation_parts(
    text: str,
    action: str,
    *,
    title: str = "",
    options: Optional[List[Dict[str, str]]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """构建 input-required 消息 parts：text 兜底 + data 供 Web 渲染按钮。"""
    parts: List[Dict[str, Any]] = [{"text": text}]
    parts.append(
        {
            "mediaType": CONFIRMATION_MEDIA_TYPE,
            "data": build_confirmation_data(
                action,
                title=title,
                options=options,
                body=body,
            ),
        }
    )
    return parts


def parse_confirmation_from_parts(
    parts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从 message parts 中解析 confirmation data。"""
    for part in parts:
        data = part.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("type") == "confirmation":
            return data
        media_type = part.get("mediaType") or part.get("media_type") or ""
        if CONFIRMATION_MEDIA_TYPE in media_type:
            return data
    return None
