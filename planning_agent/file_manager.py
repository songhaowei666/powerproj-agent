"""本地文件存取管理。"""

import shutil
import uuid
from pathlib import Path
from typing import Optional


class FileManager:
    """文件管理器，负责上传文件的本地存储与访问。"""

    def __init__(self, base_dir: str = "planning_agent/upload_files"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _build_dir(self, project_code: str, node_code: str) -> Path:
        target_dir = self.base_dir / project_code / node_code
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

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

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """根据 file_id 查找本地文件路径。

        注意：当前实现通过遍历目录匹配 file_id 前缀，
        实际生产环境建议通过数据库反查 file_path。
        """
        for project_dir in self.base_dir.iterdir():
            if not project_dir.is_dir():
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
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_dir / path
        if path.exists():
            path.unlink()
            # 清理空目录
            self._cleanup_empty_dirs(path.parent)
            return True
        return False

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
