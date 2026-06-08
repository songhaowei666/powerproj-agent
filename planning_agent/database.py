"""SQLite 数据库封装 - 项目信息与节点文件管理。"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


SEED_PROJECTS = [
    {
        "project_name": "北京西500千伏输变电工程",
        "project_code": "PRJ001",
        "voltage_level": "500kv",
        "unit_code": "01",
        "line_length": 120.5,
        "substation_capacity": 1000.0,
    },
    {
        "project_name": "天津东220千伏线路改造工程",
        "project_code": "PRJ002",
        "voltage_level": "220kv",
        "unit_code": "02",
        "line_length": 85.0,
        "substation_capacity": 500.0,
    },
    {
        "project_name": "河北南网1000千伏特高压扩建工程",
        "project_code": "PRJ003",
        "voltage_level": "1000kv",
        "unit_code": "03",
        "line_length": 320.0,
        "substation_capacity": 3000.0,
    },
    {
        "project_name": "山西太原35千伏配网升级工程",
        "project_code": "PRJ004",
        "voltage_level": "35kv",
        "unit_code": "04",
        "line_length": 45.2,
        "substation_capacity": 50.0,
    },
    {
        "project_name": "山东青岛330千伏变电站新建工程",
        "project_code": "PRJ005",
        "voltage_level": "330kv",
        "unit_code": "05",
        "line_length": 68.8,
        "substation_capacity": 1500.0,
    },
    {
        "project_name": "河南郑州10千伏配网自动化改造",
        "project_code": "PRJ006",
        "voltage_level": "10kv",
        "unit_code": "06",
        "line_length": 12.5,
        "substation_capacity": 20.0,
    },
    {
        "project_name": "湖北武汉220千伏环网工程",
        "project_code": "PRJ007",
        "voltage_level": "220kv",
        "unit_code": "07",
        "line_length": 156.0,
        "substation_capacity": 800.0,
    },
    {
        "project_name": "湖南长沙500千伏智能变电站",
        "project_code": "PRJ008",
        "voltage_level": "500kv",
        "unit_code": "08",
        "line_length": 95.3,
        "substation_capacity": 1200.0,
    },
    {
        "project_name": "江苏南京1000千伏特高压交流工程",
        "project_code": "PRJ009",
        "voltage_level": "1000kv",
        "unit_code": "09",
        "line_length": 280.0,
        "substation_capacity": 4500.0,
    },
    {
        "project_name": "浙江杭州220千伏电缆隧道工程",
        "project_code": "PRJ010",
        "voltage_level": "220kv",
        "unit_code": "10",
        "line_length": 42.0,
        "substation_capacity": 600.0,
    },
]


class ProjectDatabase:
    """项目数据库封装。"""

    def __init__(self, db_path: str = "planning_agent/planning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()
        self.seed_data()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_tables(self) -> None:
        """初始化数据表。"""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS project_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    project_code TEXT NOT NULL UNIQUE,
                    voltage_level TEXT,
                    unit_code TEXT,
                    line_length REAL,
                    substation_capacity REAL
                );

                CREATE TABLE IF NOT EXISTS project_node_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_code TEXT NOT NULL,
                    node_code TEXT NOT NULL,
                    file_id TEXT NOT NULL UNIQUE,
                    file_name TEXT,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS project_info_fts USING fts5(
                    project_name, project_code,
                    content='project_info',
                    content_rowid='id'
                );
                """
            )
            conn.commit()

    def seed_data(self) -> None:
        """内置种子数据。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM project_info"
            )
            if cursor.fetchone()["cnt"] > 0:
                return

            for proj in SEED_PROJECTS:
                conn.execute(
                    """
                    INSERT INTO project_info
                    (project_name, project_code, voltage_level, unit_code, line_length, substation_capacity)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        proj["project_name"],
                        proj["project_code"],
                        proj["voltage_level"],
                        proj["unit_code"],
                        proj["line_length"],
                        proj["substation_capacity"],
                    ),
                )
            conn.commit()

    def search_projects(
        self,
        keywords: Optional[str] = None,
        voltage_level: Optional[str] = None,
        unit_code: Optional[str] = None,
        min_line_length: Optional[float] = None,
        max_line_length: Optional[float] = None,
        min_capacity: Optional[float] = None,
        max_capacity: Optional[float] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """多条件组合查询项目。"""
        conditions = ["1=1"]
        params: List[Any] = []

        if keywords:
            conditions.append(
                "(project_name LIKE ? OR project_code LIKE ?)"
            )
            like_pattern = f"%{keywords}%"
            params.extend([like_pattern, like_pattern])

        if voltage_level:
            conditions.append("voltage_level = ?")
            params.append(voltage_level)

        if unit_code:
            conditions.append("unit_code = ?")
            params.append(unit_code)

        if min_line_length is not None:
            conditions.append("line_length >= ?")
            params.append(min_line_length)

        if max_line_length is not None:
            conditions.append("line_length <= ?")
            params.append(max_line_length)

        if min_capacity is not None:
            conditions.append("substation_capacity >= ?")
            params.append(min_capacity)

        if max_capacity is not None:
            conditions.append("substation_capacity <= ?")
            params.append(max_capacity)

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM project_info WHERE {where_clause} LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_project_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """根据项目编码查询。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM project_info WHERE project_code = ?",
                (code,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def execute_aggregate_query(self, sql: str) -> Dict[str, Any]:
        """执行聚合查询。"""
        with self._connect() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            return dict(row) if row else {}

    def add_file_record(
        self,
        project_code: str,
        node_code: str,
        file_id: str,
        file_name: str,
        file_path: str,
    ) -> None:
        """添加或更新文件记录（同名覆盖）。"""
        with self._connect() as conn:
            # 先检查同项目同节点同名文件是否存在
            cursor = conn.execute(
                "SELECT file_id FROM project_node_files WHERE project_code = ? AND node_code = ? AND file_name = ?",
                (project_code, node_code, file_name),
            )
            existing = cursor.fetchone()
            if existing:
                # 覆盖：更新 file_path 和 created_at
                conn.execute(
                    "UPDATE project_node_files SET file_path = ?, created_at = CURRENT_TIMESTAMP WHERE file_id = ?",
                    (file_path, existing["file_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO project_node_files (project_code, node_code, file_id, file_name, file_path) VALUES (?, ?, ?, ?, ?)",
                    (project_code, node_code, file_id, file_name, file_path),
                )
            conn.commit()

    def list_files(
        self,
        project_code: str,
        node_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询文件列表。"""
        with self._connect() as conn:
            if node_code:
                cursor = conn.execute(
                    "SELECT * FROM project_node_files WHERE project_code = ? AND node_code = ? ORDER BY created_at DESC",
                    (project_code, node_code),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM project_node_files WHERE project_code = ? ORDER BY created_at DESC",
                    (project_code,),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """根据 file_id 查询。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM project_node_files WHERE file_id = ?",
                (file_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_file_by_name(
        self, project_code: str, node_code: str, file_name: str
    ) -> Optional[Dict[str, Any]]:
        """根据文件名查询。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM project_node_files WHERE project_code = ? AND node_code = ? AND file_name = ?",
                (project_code, node_code, file_name),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_file_record(self, file_id: str) -> bool:
        """删除文件记录。"""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM project_node_files WHERE file_id = ?",
                (file_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
