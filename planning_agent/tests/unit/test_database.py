"""database.py 单元测试。"""

import os
import tempfile

import pytest

from planning_agent.database import ProjectDatabase, SEED_PROJECTS


@pytest.fixture
def temp_db():
    """提供临时数据库实例。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = ProjectDatabase(db_path=db_path)
    yield db
    os.unlink(db_path)


class TestDatabase:
    """数据库操作测试。"""

    def test_seed_data_loaded(self, temp_db):
        """断言种子数据 10 条全部加载。"""
        projects = temp_db.search_projects(limit=100)
        assert len(projects) == len(SEED_PROJECTS)

    def test_search_by_keywords(self, temp_db):
        """关键词模糊匹配。"""
        results = temp_db.search_projects(keywords="北京西")
        assert len(results) == 1
        assert results[0]["project_code"] == "PRJ001"

    def test_search_by_voltage_level(self, temp_db):
        """电压等级筛选（220kv 返回 3 条）。"""
        results = temp_db.search_projects(voltage_level="220kv")
        assert len(results) == 3
        codes = {r["project_code"] for r in results}
        assert codes == {"PRJ002", "PRJ007", "PRJ010"}

    def test_search_by_unit_code(self, temp_db):
        """单位编码筛选。"""
        results = temp_db.search_projects(unit_code="01")
        assert len(results) == 1
        assert results[0]["project_code"] == "PRJ001"

    def test_search_by_line_length_range(self, temp_db):
        """线路长度范围筛选。"""
        results = temp_db.search_projects(
            min_line_length=100, max_line_length=300
        )
        assert len(results) == 3
        codes = {r["project_code"] for r in results}
        assert codes == {"PRJ001", "PRJ007", "PRJ009"}

    def test_search_by_substation_capacity_range(self, temp_db):
        """变电容量范围筛选（1000-2000 MVA 返回 3 条）。"""
        results = temp_db.search_projects(
            min_capacity=1000, max_capacity=2000
        )
        assert len(results) == 3
        codes = {r["project_code"] for r in results}
        assert codes == {"PRJ001", "PRJ005", "PRJ008"}

    def test_search_combined_conditions(self, temp_db):
        """名称 + 电压等级组合查询。"""
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

    def test_get_project_by_code_not_found(self, temp_db):
        """查询不存在的编码返回 None。"""
        assert temp_db.get_project_by_code("PRJ999") is None

    def test_aggregate_query(self, temp_db):
        """SUM/COUNT 聚合查询。"""
        result = temp_db.execute_aggregate_query(
            "SELECT SUM(substation_capacity) AS total_capacity FROM project_info"
        )
        assert "total_capacity" in result
        assert result["total_capacity"] > 0

    def test_aggregate_query_empty_result(self, temp_db):
        """聚合查询无结果时返回带 None 值的字典。"""
        result = temp_db.execute_aggregate_query(
            "SELECT SUM(substation_capacity) AS total_capacity FROM project_info WHERE 1=0"
        )
        assert "total_capacity" in result
        assert result["total_capacity"] is None

    def test_file_record_cover(self, temp_db):
        """同名文件覆盖（同一记录更新 file_path，保留原 file_id）。"""
        first_id = temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="test-uuid-1",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report.pdf",
        )
        assert first_id == "test-uuid-1"
        files = temp_db.list_files("PRJ001", "001")
        assert len(files) == 1
        assert files[0]["file_path"] == "upload_files/PRJ001/001/report.pdf"

        # 覆盖同名文件
        second_id = temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="test-uuid-2",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report_v2.pdf",
        )
        assert second_id == "test-uuid-1"
        files = temp_db.list_files("PRJ001", "001")
        assert len(files) == 1
        assert files[0]["file_path"] == "upload_files/PRJ001/001/report_v2.pdf"
        assert files[0]["file_id"] == "test-uuid-1"

        # 清理
        temp_db.delete_file_record("test-uuid-1")

    def test_list_files_without_node_code(self, temp_db):
        """不带 node_code 查询项目下所有文件。"""
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="file-1",
            file_name="a.pdf",
            file_path="upload_files/PRJ001/001/a.pdf",
        )
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="002",
            file_id="file-2",
            file_name="b.pdf",
            file_path="upload_files/PRJ001/002/b.pdf",
        )

        files = temp_db.list_files("PRJ001")
        assert len(files) == 2
        names = {f["file_name"] for f in files}
        assert names == {"a.pdf", "b.pdf"}

    def test_get_file_by_id(self, temp_db):
        """根据 file_id 查询文件记录。"""
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="file-by-id",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report.pdf",
        )
        file_info = temp_db.get_file_by_id("file-by-id")
        assert file_info is not None
        assert file_info["file_name"] == "report.pdf"

        assert temp_db.get_file_by_id("not-exist") is None

    def test_get_file_by_name(self, temp_db):
        """根据文件名查询文件记录。"""
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id="file-by-name",
            file_name="report.pdf",
            file_path="upload_files/PRJ001/001/report.pdf",
        )
        file_info = temp_db.get_file_by_name("PRJ001", "001", "report.pdf")
        assert file_info is not None
        assert file_info["file_id"] == "file-by-name"

        assert temp_db.get_file_by_name("PRJ001", "001", "none.pdf") is None

    def test_delete_file_record_returns_false_when_not_found(self, temp_db):
        """删除不存在的文件记录返回 False。"""
        result = temp_db.delete_file_record("not-exist")
        assert result is False
