"""A2A message.parts 解析器。

约定格式（由主控 Agent build_task_parts 生成）：
- 第一个 text part：当前任务描述
- 以「【前置任务」开头的 text part：前置任务分段标题
- 后续 text/url part：归入当前前置分段
- raw part：用户上传文件（与前置任务无关）
- url part（主消息内）：用户提供的文件地址附件
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


UPSTREAM_MARKER_PREFIX = "【前置任务"


@dataclass
class UpstreamSection:
    """单个前置任务的注入内容。"""

    header: str
    parts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedInput:
    """解析后的 A2A 输入消息。"""

    task_query: str
    upstream_sections: List[UpstreamSection] = field(default_factory=list)
    raw_files: List[Dict[str, Any]] = field(default_factory=list)
    attachment_files: List[Dict[str, Any]] = field(default_factory=list)


def build_upstream_header(task_id: str, capability: str, name: str) -> str:
    """构建前置任务分段标题。"""
    return f"【前置任务 {task_id} - {capability} - {name}】"


def format_upstream_context(parsed: ParsedInput) -> str:
    """将前置分段格式化为可读上下文字符串。"""
    if not parsed.upstream_sections:
        return ""

    blocks: List[str] = []
    for section in parsed.upstream_sections:
        lines = [section.header]
        for part in section.parts:
            text = part.get("text", "")
            if text:
                lines.append(text)
                continue
            url = part.get("url", "")
            if url:
                filename = part.get("filename") or part.get("name") or "文件"
                lines.append(f"文件 [{filename}]: {url}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_message_parts(message: Any) -> ParsedInput:
    """从 protobuf Message 对象解析 parts。"""
    if message is None:
        return ParsedInput(task_query="")

    dict_parts: List[Dict[str, Any]] = []
    for part in getattr(message, "parts", []):
        content_type = part.WhichOneof("content")
        if content_type == "text":
            dict_parts.append({"text": part.text})
        elif content_type == "url":
            dict_parts.append(
                {
                    "url": part.url,
                    "filename": part.filename or "",
                }
            )
        elif content_type == "raw":
            dict_parts.append(
                {
                    "raw": bytes(part.raw),
                    "filename": part.filename or "unnamed",
                    "mediaType": part.media_type or "",
                }
            )
    return parse_parts_list(dict_parts)


def parse_parts_list(parts: List[Dict[str, Any]]) -> ParsedInput:
    """从 parts 字典列表解析结构化输入。"""
    task_query = ""
    upstream_sections: List[UpstreamSection] = []
    raw_files: List[Dict[str, Any]] = []
    attachment_files: List[Dict[str, Any]] = []
    current_section: Optional[UpstreamSection] = None

    for part in parts:
        normalized = _normalize_part(part)
        if not normalized:
            continue

        if "raw" in normalized:
            raw_files.append(
                {
                    "name": normalized.get("filename") or "unnamed",
                    "content": normalized["raw"],
                    "mime_type": normalized.get("mediaType", ""),
                }
            )
            continue

        text = normalized.get("text", "")
        if text:
            if text.startswith(UPSTREAM_MARKER_PREFIX):
                current_section = UpstreamSection(header=text, parts=[])
                upstream_sections.append(current_section)
                continue

            if not task_query and current_section is None:
                task_query = text
                continue

            if current_section is not None:
                current_section.parts.append(normalized)
            elif not task_query:
                task_query = text
            else:
                if upstream_sections:
                    upstream_sections[-1].parts.append(normalized)
                else:
                    task_query = f"{task_query}\n{text}"
            continue

        if "url" in normalized:
            if current_section is not None:
                current_section.parts.append(normalized)
            elif upstream_sections:
                upstream_sections[-1].parts.append(normalized)
            else:
                attachment_files.append(
                    {
                        "url": normalized["url"],
                        "name": normalized.get("filename") or "文件",
                    }
                )
            continue

    return ParsedInput(
        task_query=task_query,
        upstream_sections=upstream_sections,
        raw_files=raw_files,
        attachment_files=attachment_files,
    )


def _normalize_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将多种 part 表示统一为 text/url/raw 字典。"""
    if part.get("text"):
        return {"text": part["text"]}

    url = part.get("url", "")
    if url:
        return {
            "url": url,
            "filename": part.get("filename") or part.get("name") or "",
        }

    raw = part.get("raw")
    if raw is not None:
        if isinstance(raw, str):
            import base64

            try:
                raw_bytes = base64.b64decode(raw)
            except Exception:
                raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = bytes(raw)
        return {
            "raw": raw_bytes,
            "filename": part.get("filename") or part.get("name") or "unnamed",
            "mediaType": part.get("mediaType") or part.get("mime_type") or "",
        }

    return None
