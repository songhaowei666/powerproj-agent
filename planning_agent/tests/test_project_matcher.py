"""project_matcher.py 单元测试。"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from planning_agent.database import ProjectDatabase
from planning_agent.project_matcher import ProjectMatcher


@pytest.fixture
def temp_db():
    """提供临时数据库实例。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = ProjectDatabase(db_path=db_path)
    yield db
    os.unlink(db_path)


@pytest.fixture
def mock_llm():
    """提供 mock LLM。"""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="SELECT SUM(substation_capacity) AS total_capacity FROM project_info"
        )
    )
    return llm


class TestProjectMatcher:
    """项目匹配器测试。"""

    @pytest.mark.asyncio
    async def test_match_single_result(self, temp_db, mock_llm):
        """唯一结果直接返回，match_score=1.0。"""
        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=AsyncMock(return_value=filter_mock))
        )

        matcher = ProjectMatcher(temp_db, mock_llm)
        result = await matcher.match("北京西500千伏项目")

        assert result is not None
        assert result.project_code == "PRJ001"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_match_no_result(self, temp_db, mock_llm):
        """无匹配返回 None。"""
        filter_mock = MagicMock()
        filter_mock.keywords = "不存在的项目"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=AsyncMock(return_value=filter_mock))
        )

        matcher = ProjectMatcher(temp_db, mock_llm)
        result = await matcher.match("某某不存在的项目")

        assert result is None
