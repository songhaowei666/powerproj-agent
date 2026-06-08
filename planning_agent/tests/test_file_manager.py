"""file_manager.py 单元测试。"""

import tempfile

import pytest

from planning_agent.file_manager import FileManager


@pytest.fixture
def temp_fm():
    """提供临时文件管理器。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileManager(base_dir=tmpdir)


class TestFileManager:
    """文件管理器测试。"""

    def test_save_and_get_file(self, temp_fm):
        """保存后能通过路径读取。"""
        file_id = temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="test.txt",
            content_bytes=b"hello world",
        )
        assert file_id is not None

        path = temp_fm.get_file_path_by_location("PRJ001", "001", "test.txt")
        assert path is not None
        assert path.read_bytes() == b"hello world"

    def test_delete_file(self, temp_fm):
        """删除后文件不存在。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="del.txt",
            content_bytes=b"delete me",
        )
        result = temp_fm.delete_file_by_location("PRJ001", "001", "del.txt")
        assert result is True

        path = temp_fm.get_file_path_by_location("PRJ001", "001", "del.txt")
        assert path is None

    def test_build_download_url(self, temp_fm):
        """URL 格式正确。"""
        url = temp_fm.build_download_url("uuid-123", "http://localhost:8001")
        assert url == "http://localhost:8001/files/uuid-123"
