"""Planning Agent 单元测试。"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from planning_agent.database import ProjectDatabase, SEED_PROJECTS
from planning_agent.file_manager import FileManager
from planning_agent.models import MatchedProject, PlanningState
from planning_agent.project_matcher import ProjectMatcher, ProjectFilter


# ---------- Fixtures ----------


@pytest.fixture
def temp_db():
    """提供临时数据库实例。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = ProjectDatabase(db_path=db_path)
    yield db
    os.unlink(db_path)


@pytest.fixture
def temp_fm():
    """提供临时文件管理器。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileManager(base_dir=tmpdir)


@pytest.fixture
def mock_llm():
    """提供 mock LLM。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="SELECT SUM(substation_capacity) AS 变电容量总和 FROM project_info"))
    return llm


# ---------- Database Tests ----------


class TestDatabase:
    """数据库操作测试。"""

    def test_seed_data_loaded(self, temp_db):
        """种子数据已正确加载。"""
        projects = temp_db.search_projects(limit=100)
        assert len(projects) == len(SEED_PROJECTS)

    def test_search_by_keywords(self, temp_db):
        """关键词搜索。"""
        results = temp_db.search_projects(keywords="北京西")
        assert len(results) == 1
        assert results[0]["project_code"] == "PRJ001"

    def test_search_by_voltage_level(self, temp_db):
        """按电压等级筛选。"""
        results = temp_db.search_projects(voltage_level="220kv")
        assert len(results) == 3
        codes = {r["project_code"] for r in results}
        assert codes == {"PRJ002", "PRJ007", "PRJ010"}

    def test_search_by_unit_code(self, temp_db):
        """按单位编码筛选。"""
        results = temp_db.search_projects(unit_code="01")
        assert len(results) == 1
        assert results[0]["project_code"] == "PRJ001"

    def test_search_by_line_length_range(self, temp_db):
        """按线路长度范围筛选。"""
        results = temp_db.search_projects(
            min_line_length=100, max_line_length=300
        )
        assert len(results) == 3
        codes = {r["project_code"] for r in results}
        assert codes == {"PRJ001", "PRJ007", "PRJ009"}

    def test_search_combined_conditions(self, temp_db):
        """组合条件查询。"""
        results = temp_db.search_projects(
            keywords="河北",
            voltage_level="1000kv",
        )
        assert len(results) == 1
        assert results[0]["project_code"] == "PRJ003"

    def test_get_project_by_code(self, temp_db):
        """根据编码查询。"""
        proj = temp_db.get_project_by_code("PRJ005")
        assert proj is not None
        assert proj["project_name"] == "山东青岛330千伏变电站新建工程"

    def test_aggregate_query(self, temp_db):
        """聚合查询。"""
        result = temp_db.execute_aggregate_query(
            "SELECT SUM(substation_capacity) AS total_capacity FROM project_info"
        )
        assert "total_capacity" in result
        assert result["total_capacity"] > 0

    def test_file_record_crud(self, temp_db):
        """文件记录增删改查。"""
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="test-uuid-1",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report.pdf",
        )
        files = temp_db.list_files("PRJ001", "001")
        assert len(files) == 1
        assert files[0]["file_name"] == "report.pdf"

        # 覆盖同名文件
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="test-uuid-2",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report_v2.pdf",
        )
        files = temp_db.list_files("PRJ001", "001")
        assert len(files) == 1
        assert files[0]["file_path"] == "upload_files/PRJ001/001/report_v2.pdf"

        # 删除
        temp_db.delete_file_record("test-uuid-1")
        files = temp_db.list_files("PRJ001")
        assert len(files) == 0


# ---------- File Manager Tests ----------


class TestFileManager:
    """文件管理器测试。"""

    def test_save_and_get_file(self, temp_fm):
        """保存和读取文件。"""
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
        """删除文件。"""
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
        """构造下载 URL。"""
        url = temp_fm.build_download_url("uuid-123", "http://localhost:8001")
        assert url == "http://localhost:8001/files/uuid-123"


# ---------- Project Matcher Tests ----------


class TestProjectMatcher:
    """项目匹配器测试。"""

    @pytest.mark.asyncio
    async def test_match_single_result(self, temp_db, mock_llm):
        """唯一结果直接返回。"""
        # Mock filter 返回精确匹配条件
        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(return_value=MagicMock(
            ainvoke=AsyncMock(return_value=filter_mock)
        ))

        matcher = ProjectMatcher(temp_db, mock_llm)
        result = await matcher.match("北京西500千伏项目")

        assert result is not None
        assert result.project_code == "PRJ001"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_match_no_result(self, temp_db, mock_llm):
        """无匹配结果返回 None。"""
        filter_mock = MagicMock()
        filter_mock.keywords = "不存在的项目"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(return_value=MagicMock(
            ainvoke=AsyncMock(return_value=filter_mock)
        ))

        matcher = ProjectMatcher(temp_db, mock_llm)
        result = await matcher.match("某某不存在的项目")

        assert result is None


# ---------- Models Tests ----------


class TestModels:
    """Pydantic 模型测试。"""

    def test_matched_project_serialization(self):
        """MatchedProject 序列化。"""
        proj = MatchedProject(
            project_name="测试项目",
            project_code="PRJ_TEST",
            voltage_level="220kv",
            unit_code="01",
            line_length=100.0,
            substation_capacity=500.0,
            match_score=0.95,
        )
        data = proj.model_dump()
        assert data["project_code"] == "PRJ_TEST"
        assert data["voltage_level"] == "220kv"
        assert data["unit_code"] == "01"

    def test_planning_state_defaults(self):
        """PlanningState 默认值。"""
        state = PlanningState()
        assert state.intent == "unknown"
        assert state.status == "pending"
        assert state.project_confirmed is False


# ---------- Graph Integration Tests ----------


class TestGraphNodes:
    """Graph 节点逻辑测试。"""

    def test_graph_compiles(self, mock_llm, temp_db, temp_fm):
        """Graph 能正常编译。"""
        from planning_agent.graph import build_planning_graph
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        assert graph is not None
