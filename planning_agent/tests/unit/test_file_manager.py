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

    def test_save_overwrites_existing_file(self, temp_fm):
        """同名文件保存后覆盖旧内容。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="test.txt",
            content_bytes=b"old content",
        )
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="test.txt",
            content_bytes=b"new content",
        )

        path = temp_fm.get_file_path_by_location("PRJ001", "001", "test.txt")
        assert path.read_bytes() == b"new content"

    def test_delete_file_by_location(self, temp_fm):
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

    def test_delete_file_by_location_not_found(self, temp_fm):
        """删除不存在的文件返回 False。"""
        result = temp_fm.delete_file_by_location("PRJ001", "001", "none.txt")
        assert result is False

    def test_delete_file_and_cleanup_empty_dirs(self, temp_fm):
        """删除文件后自动清理空目录。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="cleanup.txt",
            content_bytes=b"cleanup",
        )
        rel_path = "PRJ001/001/cleanup.txt"
        result = temp_fm.delete_file(rel_path)
        assert result is True

        # 目录应被清理
        assert not (temp_fm.base_dir / "PRJ001").exists()

    def test_delete_file_not_found(self, temp_fm):
        """删除不存在的文件路径返回 False。"""
        result = temp_fm.delete_file("PRJ001/001/none.txt")
        assert result is False

    def test_get_file_path_finds_existing_file(self, temp_fm):
        """get_file_path 能找到已保存的文件。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="find_me.txt",
            content_bytes=b"found",
        )
        path = temp_fm.get_file_path("any-id")
        assert path is not None
        assert path.read_bytes() == b"found"

    def test_get_file_path_returns_none_when_empty(self, temp_fm):
        """空目录下 get_file_path 返回 None。"""
        assert temp_fm.get_file_path("any-id") is None

    def test_get_file_path_skips_non_directory_entries(self, temp_fm):
        """get_file_path 跳过 base_dir 下的非目录文件。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="target.txt",
            content_bytes=b"target",
        )
        # 在 base_dir 下放置一个普通文件，应被跳过
        (temp_fm.base_dir / "not_a_dir.txt").write_text("ignore me")
        path = temp_fm.get_file_path("any-id")
        assert path is not None
        assert path.read_bytes() == b"target"

    def test_get_file_path_skips_non_directory_node_entries(self, temp_fm):
        """get_file_path 跳过 project_dir 下的非目录文件。"""
        # 创建 project_dir 但下面只有普通文件，没有节点目录
        (temp_fm.base_dir / "PRJ001").mkdir(parents=True, exist_ok=True)
        (temp_fm.base_dir / "PRJ001" / "not_a_node.txt").write_text("ignore me")
        path = temp_fm.get_file_path("any-id")
        assert path is None

    def test_delete_file_handles_cleanup_oserror(self, temp_fm):
        """删除文件后清理空目录遇到 OSError 不抛出。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="x.txt",
            content_bytes=b"x",
        )

        # 模拟 rmdir 抛出 OSError，验证异常被吞掉
        import pathlib
        original_rmdir = pathlib.Path.rmdir

        def _failing_rmdir(self):
            raise OSError("permission denied")

        pathlib.Path.rmdir = _failing_rmdir
        try:
            result = temp_fm.delete_file_by_location("PRJ001", "001", "x.txt")
            assert result is True
        finally:
            pathlib.Path.rmdir = original_rmdir

    def test_delete_file_stops_cleanup_when_dir_not_empty(self, temp_fm):
        """删除文件后目录不为空时停止清理。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="keep.txt",
            content_bytes=b"keep",
        )
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="remove.txt",
            content_bytes=b"remove",
        )
        result = temp_fm.delete_file_by_location("PRJ001", "001", "remove.txt")
        assert result is True
        # 目录应保留，因为还有 keep.txt
        assert (temp_fm.base_dir / "PRJ001" / "001" / "keep.txt").exists()

    def test_build_download_url(self, temp_fm):
        """URL 格式正确。"""
        url = temp_fm.build_download_url("uuid-123", "http://localhost:8001")
        assert url == "http://localhost:8001/files/uuid-123"

    def test_save_staging_and_commit(self, temp_fm):
        """暂存上传后可 commit 到项目节点并保留 file_id。"""
        file_id = temp_fm.save_staging_file(b"staging bytes", "stage.pdf")
        assert temp_fm.get_staging_file_path(file_id) is not None

        committed_id, target_path = temp_fm.commit_staging_file(
            file_id, "PRJ001", "001"
        )
        assert committed_id == file_id
        assert target_path.read_bytes() == b"staging bytes"
        assert temp_fm.get_staging_file_path(file_id) is None

    def test_extract_file_id_from_url(self):
        """从 files URL 提取 file_id。"""
        file_id = FileManager.extract_file_id_from_url(
            "http://localhost:8001/files/abc-123"
        )
        assert file_id == "abc-123"

    def test_resolve_download_path_relative(self, temp_fm):
        """相对路径（project/node/file）可正确解析。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="report.pdf",
            content_bytes=b"report content",
        )
        path = temp_fm.resolve_download_path(
            "any-id", "PRJ001/001/report.pdf"
        )
        assert path is not None
        assert path.read_bytes() == b"report content"

    def test_resolve_download_path_legacy_prefix(self, temp_fm):
        """兼容含 upload_files 前缀的历史 file_path。"""
        temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="legacy.pdf",
            content_bytes=b"legacy content",
        )
        legacy_path = f"{temp_fm.base_dir}/PRJ001/001/legacy.pdf"
        path = temp_fm.resolve_download_path("legacy-id", legacy_path)
        assert path is not None
        assert path.read_bytes() == b"legacy content"
