"""规划 Agent 文件输入解析：支持 staging URL 引用、外部 URL 与 raw 二进制。"""

from __future__ import annotations

import base64
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import unquote, urlparse

import httpx

from a2a_message_parser.parser import ParsedInput

if TYPE_CHECKING:
    from planning_agent.file_manager import FileManager

DEFAULT_DOWNLOAD_TIMEOUT = 60.0


def _filename_from_url(url: str) -> str:
    """从 URL 路径推断文件名。"""
    path = unquote(urlparse(url).path or "")
    name = PurePosixPath(path).name
    return name or "文件"


def _normalize_pending_item(
    name: str,
    content: bytes,
    mime_type: str = "",
) -> Dict[str, Any]:
    """转为 PlanningState 可序列化的 pending_files 条目（legacy raw 路径）。"""
    return {
        "name": name or "unnamed",
        "content": base64.b64encode(content).decode(),
        "mime_type": mime_type,
    }


def _normalize_staging_pending_item(
    name: str,
    staging_file_id: str,
    mime_type: str = "",
) -> Dict[str, Any]:
    """暂存 file_id 引用，execute_action 时 commit 到项目节点。"""
    item: Dict[str, Any] = {
        "name": name or "unnamed",
        "staging_file_id": staging_file_id,
    }
    if mime_type:
        item["mime_type"] = mime_type
    return item


async def resolve_pending_files(
    parsed: ParsedInput,
    *,
    file_manager: Optional["FileManager"] = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> List[Dict[str, Any]]:
    """合并 raw 附件与 URL 附件为 pending_files。"""
    pending: List[Dict[str, Any]] = []

    for item in parsed.raw_files:
        content = item.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            content_bytes = base64.b64decode(content)
        else:
            content_bytes = bytes(content)
        pending.append(
            _normalize_pending_item(
                str(item.get("name") or "unnamed"),
                content_bytes,
                str(item.get("mime_type") or ""),
            )
        )

    if not parsed.attachment_files:
        return pending

    external_urls: List[Dict[str, Any]] = []
    for item in parsed.attachment_files:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        filename = str(item.get("name") or "").strip() or _filename_from_url(url)
        if file_manager is not None:
            file_id = file_manager.extract_file_id_from_url(url)
            if file_id and file_manager.get_staging_file_path(file_id) is not None:
                pending.append(_normalize_staging_pending_item(filename, file_id))
                continue
        external_urls.append({"url": url, "name": filename})

    if not external_urls:
        return pending

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for item in external_urls:
            url = item["url"]
            response = await client.get(url)
            response.raise_for_status()
            filename = item["name"]
            mime_type = response.headers.get("content-type", "")
            pending.append(
                _normalize_pending_item(filename, response.content, mime_type)
            )

    return pending
