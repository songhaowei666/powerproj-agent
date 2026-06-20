"""file_input 单元测试。"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_message_parser.parser import ParsedInput
from planning_agent.file_input import resolve_pending_files
from planning_agent.file_manager import FileManager


@pytest.mark.asyncio
async def test_resolve_pending_files_from_external_url():
    """外部 URL 附件应下载为 pending_files 字节内容。"""
    mock_response = MagicMock()
    mock_response.content = b"pdf-bytes"
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    parsed = ParsedInput(
        task_query="上传文件",
        attachment_files=[{"url": "http://example.com/design.pdf", "name": "design.pdf"}],
    )
    with patch("planning_agent.file_input.httpx.AsyncClient", return_value=mock_client):
        pending = await resolve_pending_files(parsed)

    assert len(pending) == 1
    assert pending[0]["name"] == "design.pdf"
    assert base64.b64decode(pending[0]["content"]) == b"pdf-bytes"


@pytest.mark.asyncio
async def test_resolve_pending_files_from_local_staging_url(tmp_path):
    """规划 Agent 本地 /files/{id} URL 应解析为 staging_file_id 引用。"""
    fm = FileManager(base_dir=str(tmp_path))
    file_id = fm.save_staging_file(b"local-content", "design.pdf")

    parsed = ParsedInput(
        task_query="上传文件",
        attachment_files=[
            {
                "url": f"http://localhost:8001/files/{file_id}",
                "name": "design.pdf",
            }
        ],
    )
    pending = await resolve_pending_files(parsed, file_manager=fm)

    assert len(pending) == 1
    assert pending[0]["staging_file_id"] == file_id
    assert pending[0]["name"] == "design.pdf"
    assert "content" not in pending[0]
