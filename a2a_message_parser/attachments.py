"""主消息附件 parts 构建工具。"""

from typing import Any, Dict, List

from a2a_message_parser.parser import ParsedInput


def build_attachment_parts(parsed: ParsedInput) -> List[Dict[str, Any]]:
    """将当前消息中的 raw/url 附件转为可转发的 parts 字典列表。"""
    parts: List[Dict[str, Any]] = []
    for item in parsed.raw_files:
        part: Dict[str, Any] = {
            "filename": item.get("name") or "unnamed",
            "raw": item.get("content"),
        }
        mime_type = item.get("mime_type") or ""
        if mime_type:
            part["mediaType"] = mime_type
        parts.append(part)

    for item in parsed.attachment_files:
        url = item.get("url", "")
        if not url:
            continue
        parts.append(
            {
                "url": url,
                "filename": item.get("name") or "文件",
            }
        )
    return parts
