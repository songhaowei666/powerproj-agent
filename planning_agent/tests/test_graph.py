"""graph.py 单元测试。"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.models import PlanningState


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
    llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="SELECT COUNT(*) AS cnt FROM project_info"
        )
    )
    return llm


class TestGraph:
    """Graph 测试。"""

    def test_graph_compiles(self, mock_llm, temp_db, temp_fm):
        """LangGraph 编译成功。"""
        from planning_agent.graph import build_planning_graph

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_parse_intent_aggregate(self, mock_llm, temp_db, temp_fm):
        """聚合查询 intent=query_project，跳过 match/confirm。"""
        # mock parse_intent 返回 query_project
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})
        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,  # parse_intent
                MagicMock(  # aggregate sql generation
                    ainvoke=AsyncMock(
                        return_value=MagicMock(
                            content="SELECT COUNT(*) AS cnt FROM project_info"
                        )
                    )
                ),
            ]
        )

        from planning_agent.graph import build_planning_graph

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        # 聚合查询直接执行，不经过项目匹配
        state = PlanningState(query="一共有多少个项目")
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-agg-1"}}
        )

        assert result is not None
        assert result.get("intent") == "query_project"
        assert result.get("matched_project") is None
        assert result.get("project_confirmed") is True
