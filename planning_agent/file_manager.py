"""本地文件存取管理。"""

import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


STAGING_DIR_NAME = "_staging"


class FileManager:
    """文件管理器，负责上传文件的本地存储与访问。"""

    def __init__(self, base_dir: str = "planning_agent/upload_files"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.base_dir / STAGING_DIR_NAME
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def _build_dir(self, project_code: str, node_code: str) -> Path:
        target_dir = self.base_dir / project_code / node_code
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    @staticmethod
    def extract_file_id_from_url(url: str) -> Optional[str]:
        """从 /files/{file_id} 形式的 URL 提取 file_id。"""
        path = urlparse(url).path or ""
        marker = "/files/"
        if marker not in path:
            return None
        file_id = path.split(marker, 1)[-1].split("/")[0].strip()
        return file_id or None

    def save_staging_file(self, content_bytes: bytes, file_name: str) -> str:
        """保存暂存上传文件，返回 file_id。"""
        file_id = str(uuid.uuid4())
        target_dir = self.staging_dir / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file_name).name or "unnamed"
        (target_dir / safe_name).write_bytes(content_bytes)
        return file_id

    def get_staging_file_path(self, file_id: str) -> Optional[Path]:
        """获取暂存文件路径。"""
        staging_root = self.staging_dir / file_id
        if not staging_root.is_dir():
            return None
        for item in staging_root.iterdir():
            if item.is_file():
                return item
        return None

    def read_staging_file(self, file_id: str) -> Optional[Tuple[str, bytes]]:
        """读取暂存文件，返回 (文件名, 内容)。"""
        path = self.get_staging_file_path(file_id)
        if path is None:
            return None
        return path.name, path.read_bytes()

    def commit_staging_file(
        self,
        file_id: str,
        project_code: str,
        node_code: str,
    ) -> Tuple[str, Path]:
        """将暂存文件写入项目节点目录，保留原 file_id。"""
        staging_info = self.read_staging_file(file_id)
        if staging_info is None:
            raise FileNotFoundError(f"暂存文件不存在: {file_id}")
        file_name, content_bytes = staging_info
        target_dir = self._build_dir(project_code, node_code)
        target_path = target_dir / file_name
        target_path.write_bytes(content_bytes)
        shutil.rmtree(self.staging_dir / file_id, ignore_errors=True)
        return file_id, target_path

    def save_uploaded_file(
        self,
        project_code: str,
        node_code: str,
        file_name: str,
        content_bytes: bytes,
    ) -> str:
        """保存上传文件，返回 file_id (UUID)。

        同名文件直接覆盖。
        """
        file_id = str(uuid.uuid4())
        target_dir = self._build_dir(project_code, node_code)
        file_path = target_dir / file_name
        file_path.write_bytes(content_bytes)
        return file_id

    def _resolve_stored_path(self, db_file_path: str) -> Optional[Path]:
        """解析数据库中存储的 file_path 为本地绝对路径。"""
        stored = Path(db_file_path)
        if stored.is_absolute() and stored.exists():
            return stored
        if stored.exists():
            return stored

        under_base = self.base_dir / db_file_path
        if under_base.exists():
            return under_base

        # 兼容历史记录：file_path 含 upload_files 前缀
        base_name = self.base_dir.name
        parts = stored.parts
        if base_name in parts:
            idx = parts.index(base_name)
            rel = Path(*parts[idx + 1 :])
            legacy = self.base_dir / rel
            if legacy.exists():
                return legacy
        return None

    def resolve_download_path(
        self,
        file_id: str,
        db_file_path: Optional[str] = None,
    ) -> Optional[Path]:
        """按 file_id 解析可下载路径（正式目录或暂存目录）。"""
        if db_file_path:
            path = self._resolve_stored_path(db_file_path)
            if path is not None:
                return path

        return self.get_staging_file_path(file_id)

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """根据 file_id 查找本地文件路径。

        注意：当前实现通过遍历目录匹配 file_id 前缀，
        实际生产环境建议通过数据库反查 file_path。
        """
        for project_dir in self.base_dir.iterdir():
            if not project_dir.is_dir() or project_dir.name == STAGING_DIR_NAME:
                continue
            for node_dir in project_dir.iterdir():
                if not node_dir.is_dir():
                    continue
                for f in node_dir.iterdir():
                    if f.is_file():
                        # file_id 不直接编码在文件名中，实际通过数据库查
                        return f
        return None

    def get_file_path_by_location(
        self, project_code: str, node_code: str, file_name: str
    ) -> Optional[Path]:
        """根据项目编码、节点编码、文件名获取文件路径。"""
        file_path = self.base_dir / project_code / node_code / file_name
        if file_path.exists():
            return file_path
        return None

    def delete_file(self, file_path: str) -> bool:
        """删除本地文件。

        Args:
            file_path: 文件相对路径或绝对路径
        """
        path = self._resolve_stored_path(file_path)
        if path is None:
            return False
        path.unlink()
        # 清理空目录
        self._cleanup_empty_dirs(path.parent)
        return True

    def delete_file_by_location(
        self, project_code: str, node_code: str, file_name: str
    ) -> bool:
        """根据位置删除文件。"""
        path = self.base_dir / project_code / node_code / file_name
        if path.exists():
            path.unlink()
            self._cleanup_empty_dirs(path.parent)
            return True
        return False

    def _cleanup_empty_dirs(self, path: Path) -> None:
        """递归清理空目录。"""
        try:
            while path != self.base_dir:
                if not any(path.iterdir()):
                    path.rmdir()
                    path = path.parent
                else:
                    break
        except OSError:
            pass

    def build_download_url(self, file_id: str, base_url: str) -> str:
        """构造文件下载 URL。"""
        base = base_url.rstrip("/")
        return f"{base}/files/{file_id}"
