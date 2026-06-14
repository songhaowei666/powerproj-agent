"""graph.py 单元测试。"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Command

from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.graph import build_planning_graph, _is_aggregate_query
from planning_agent.models import MatchedProject, PlanningState


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
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_parse_intent_aggregate(self, mock_llm, temp_db, temp_fm):
        """聚合查询 intent=query_project，跳过 match/confirm。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})
        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(
                    ainvoke=AsyncMock(
                        return_value=MagicMock(
                            content="SELECT COUNT(*) AS cnt FROM project_info"
                        )
                    )
                ),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(query="一共有多少个项目")
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-agg-1"}}
        )

        assert result is not None
        assert result.get("intent") == "query_project"
        assert result.get("matched_project") is None
        assert result.get("project_confirmed") is True
        assert result.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_parse_intent_aggregate_fallback_keywords(self, mock_llm, temp_db, temp_fm):
        """聚合查询 LLM 异常时回退到关键词匹配。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        aggregate_structured = MagicMock()
        aggregate_structured.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                aggregate_structured,
                MagicMock(
                    ainvoke=AsyncMock(
                        return_value=MagicMock(
                            content="SELECT COUNT(*) AS cnt FROM project_info"
                        )
                    )
                ),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(query="所有项目变电容量的总和")
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-agg-fallback-1"}}
        )

        assert result is not None
        assert result.get("intent") == "query_project"
        assert result.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_parse_intent_detail_query(self, mock_llm, temp_db, temp_fm):
        """明细查询需要匹配并确认项目。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        # 第一轮：触发项目确认 interrupt
        state = PlanningState(query="查一下北京西500千伏项目")
        with patch(
            "planning_agent.graph._is_aggregate_query", return_value=False
        ):
            result = await graph.ainvoke(
                state, config={"configurable": {"thread_id": "test-detail-1"}}
            )

        # graph 节点不设置 input_required，该状态由 executor 设置
        assert result.get("matched_project") is not None
        assert result.get("matched_project").project_code == "PRJ001"
        # 检查 graph 处于中断状态
        graph_state = await graph.aget_state(
            config={"configurable": {"thread_id": "test-detail-1"}}
        )
        assert graph_state.next is not None

        # 第二轮：恢复并确认
        with patch(
            "planning_agent.graph._is_aggregate_query", return_value=False
        ):
            result = await graph.ainvoke(
                Command(resume="是的"),
                config={"configurable": {"thread_id": "test-detail-1"}},
            )
        assert result.get("status") == "completed"
        assert "PRJ001" in result.get("result_text", "")

    @pytest.mark.asyncio
    async def test_confirm_project_negative(self, mock_llm, temp_db, temp_fm):
        """用户否定项目匹配后任务失败。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(query="查一下北京西500千伏项目")
        with patch(
            "planning_agent.graph._is_aggregate_query", return_value=False
        ):
            await graph.ainvoke(
                state, config={"configurable": {"thread_id": "test-neg-1"}}
            )

        with patch(
            "planning_agent.graph._is_aggregate_query", return_value=False
        ):
            result = await graph.ainvoke(
                Command(resume="不是"),
                config={"configurable": {"thread_id": "test-neg-1"}},
            )
        assert result.get("status") == "failed"
        assert "匹配失败" in result.get("result_text", "")

    @pytest.mark.asyncio
    async def test_upload_file_flow(self, mock_llm, temp_db, temp_fm):
        """上传文件流程。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "upload_file"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(
            query="上传可研设计文件到北京西项目",
            pending_files=[{"name": "design.pdf", "content": b"file content"}],
        )
        await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-upload-1"}}
        )

        result = await graph.ainvoke(
            Command(resume="是的"),
            config={"configurable": {"thread_id": "test-upload-1"}},
        )
        assert result.get("status") == "completed"
        assert "成功上传" in result.get("result_text", "")

        files = temp_db.list_files("PRJ001", "001")
        assert len(files) == 1
        assert files[0]["file_name"] == "design.pdf"

    @pytest.mark.asyncio
    async def test_download_file_flow(self, mock_llm, temp_db, temp_fm):
        """下载文件流程。"""
        # 先准备一条文件记录和物理文件
        file_id = temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="report.pdf",
            content_bytes=b"report content",
        )
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id=file_id,
            file_name="report.pdf",
            file_path=str(temp_fm.base_dir / "PRJ001" / "001" / "report.pdf"),
        )

        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "download_file"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(query="下载北京西项目的可研设计文件")
        await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-download-1"}}
        )

        result = await graph.ainvoke(
            Command(resume="是的"),
            config={"configurable": {"thread_id": "test-download-1"}},
        )
        assert result.get("status") == "completed"
        artifacts = result.get("artifacts", [])
        file_artifacts = [
            a for a in artifacts if a.get("type") == "file"
        ]
        assert len(file_artifacts) == 1
        assert file_id in file_artifacts[0]["file"]["uri"]

    @pytest.mark.asyncio
    async def test_delete_file_flow(self, mock_llm, temp_db, temp_fm):
        """删除文件流程。"""
        file_id = temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="to_delete.pdf",
            content_bytes=b"delete me",
        )
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id=file_id,
            file_name="to_delete.pdf",
            file_path=str(temp_fm.base_dir / "PRJ001" / "001" / "to_delete.pdf"),
        )

        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "delete_file"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(
            query="删除北京西项目可研设计节点的to_delete.pdf",
            target_file_name="to_delete.pdf",
            node_code="001",
        )
        # 第一轮：项目确认 interrupt
        await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-delete-1"}}
        )
        # 第二轮：确认项目
        await graph.ainvoke(
            Command(resume="是的"),
            config={"configurable": {"thread_id": "test-delete-1"}},
        )
        # 第三轮：确认删除
        result = await graph.ainvoke(
            Command(resume="确认删除"),
            config={"configurable": {"thread_id": "test-delete-1"}},
        )
        assert result.get("status") == "completed"
        assert "已删除" in result.get("result_text", "")
        assert temp_db.get_file_by_id(file_id) is None

    @pytest.mark.asyncio
    async def test_delete_file_cancelled(self, mock_llm, temp_db, temp_fm):
        """删除文件时用户取消。"""
        file_id = temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="keep.pdf",
            content_bytes=b"keep me",
        )
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id=file_id,
            file_name="keep.pdf",
            file_path=str(temp_fm.base_dir / "PRJ001" / "001" / "keep.pdf"),
        )

        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "delete_file"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)

        state = PlanningState(
            query="删除北京西项目可研设计节点的keep.pdf",
            target_file_name="keep.pdf",
            node_code="001",
        )
        await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-delete-cancel-1"}}
        )
        await graph.ainvoke(
            Command(resume="是的"),
            config={"configurable": {"thread_id": "test-delete-cancel-1"}},
        )
        result = await graph.ainvoke(
            Command(resume="取消"),
            config={"configurable": {"thread_id": "test-delete-cancel-1"}},
        )
        assert result.get("status") == "completed"
        assert "已取消删除" in result.get("result_text", "")
        assert temp_db.get_file_by_id(file_id) is not None

    @pytest.mark.asyncio
    async def test_unknown_intent(self, mock_llm, temp_db, temp_fm):
        """未知意图导致任务失败。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "unknown"})
        mock_llm.with_structured_output = MagicMock(
            return_value=intent_mock
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(query="随便说点什么")
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-unknown-1"}}
        )
        assert result.get("status") == "failed"
        assert "无法理解" in result.get("result_text", "")

    @pytest.mark.asyncio
    async def test_parse_intent_exception_fallback(self, mock_llm, temp_db, temp_fm):
        """parse_intent LLM 异常时 intent 回退为 unknown。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_llm.with_structured_output = MagicMock(return_value=intent_mock)

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(query="查一下项目")
        result = await graph.ainvoke(
            state, config={"configurable": {"thread_id": "test-intent-exception-1"}}
        )
        assert result.get("status") == "failed"
        assert result.get("intent") == "unknown"

    @pytest.mark.asyncio
    async def test_match_project_no_result(self, mock_llm, temp_db, temp_fm):
        """未匹配到项目时任务失败。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "不存在的项目"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(query="查一下不存在的项目")
        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            result = await graph.ainvoke(
                state, config={"configurable": {"thread_id": "test-no-match-1"}}
            )
        assert result.get("status") == "failed"
        assert "未找到匹配的项目" in result.get("result_text", "")

    @pytest.mark.asyncio
    async def test_confirm_project_vague_response(self, mock_llm, temp_db, temp_fm):
        """用户模糊回答时 project_confirmed 保持 False。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(query="查一下北京西500千伏项目")
        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            await graph.ainvoke(
                state, config={"configurable": {"thread_id": "test-vague-1"}}
            )
        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            result = await graph.ainvoke(
                Command(resume="不知道"),
                config={"configurable": {"thread_id": "test-vague-1"}},
            )
        assert result.get("project_confirmed") is False

    @pytest.mark.asyncio
    async def test_detail_query_project_not_found_in_db(self, mock_llm, temp_db, temp_fm):
        """明细查询时数据库中找不到项目详情。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        # 删除数据库中的项目，使 get_project_by_code 返回 None
        temp_db.delete_file_record = MagicMock()
        with patch.object(temp_db, "get_project_by_code", return_value=None):
            state = PlanningState(query="查一下北京西500千伏项目")
            with patch("planning_agent.graph._is_aggregate_query", return_value=False):
                await graph.ainvoke(
                    state, config={"configurable": {"thread_id": "test-detail-missing-1"}}
                )
            with patch("planning_agent.graph._is_aggregate_query", return_value=False):
                result = await graph.ainvoke(
                    Command(resume="是的"),
                    config={"configurable": {"thread_id": "test-detail-missing-1"}},
                )
        assert result.get("status") == "failed"
        assert "未找到项目详细信息" in result.get("result_text", "")

    @pytest.mark.asyncio
    async def test_upload_file_no_matched_project(self, mock_llm, temp_db, temp_fm):
        """上传文件未匹配项目时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="上传文件",
            intent="upload_file",
            node_code="001",
            pending_files=[{"name": "x.pdf", "content": b"x"}],
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未匹配到项目" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_upload_file_no_node_code(self, mock_llm, temp_db, temp_fm):
        """上传文件未指定节点编码时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="上传文件到北京西项目",
            intent="upload_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            pending_files=[{"name": "x.pdf", "content": b"x"}],
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未指定节点编码" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_upload_file_no_pending_files(self, mock_llm, temp_db, temp_fm):
        """上传文件无待上传文件时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="上传文件到北京西项目",
            intent="upload_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            node_code="001",
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未检测到上传的文件内容" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_download_file_no_matched_project(self, mock_llm, temp_db, temp_fm):
        """下载文件未匹配项目时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="下载文件",
            intent="download_file",
            node_code="001",
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未匹配到项目" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_download_file_no_files(self, mock_llm, temp_db, temp_fm):
        """下载文件时项目下无文件。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="下载北京西项目文件",
            intent="download_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            node_code="001",
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        # execute_action 节点只设置 result_text，status 由 finalize 设置
        assert "暂无文件" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_delete_file_no_matched_project(self, mock_llm, temp_db, temp_fm):
        """删除文件未匹配项目时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="删除文件",
            intent="delete_file",
            node_code="001",
            target_file_name="x.pdf",
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未匹配到项目" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_delete_file_by_id(self, mock_llm, temp_db, temp_fm):
        """通过 target_file_id 删除文件。"""
        file_id = temp_fm.save_uploaded_file(
            project_code="PRJ001",
            node_code="001",
            file_name="by_id.pdf",
            content_bytes=b"delete by id",
        )
        temp_db.add_file_record(
            project_code="PRJ001",
            node_code="001",
            file_id=file_id,
            file_name="by_id.pdf",
            file_path=str(temp_fm.base_dir / "PRJ001" / "001" / "by_id.pdf"),
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="删除文件",
            intent="delete_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            target_file_id=file_id,
            delete_confirmed=True,
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        # execute_action 节点只设置 result_text，status 由 finalize 设置
        assert "已删除" in (result.result_text or "")
        assert temp_db.get_file_by_id(file_id) is None

    @pytest.mark.asyncio
    async def test_delete_file_no_target(self, mock_llm, temp_db, temp_fm):
        """删除文件未指定目标时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="删除文件",
            intent="delete_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            delete_confirmed=True,
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未指定要删除的文件" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, mock_llm, temp_db, temp_fm):
        """删除文件在数据库中找不到时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="删除文件",
            intent="delete_file",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
            node_code="001",
            target_file_name="not_exist.pdf",
            delete_confirmed=True,
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未找到要删除的文件" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_finalize_no_result_text(self, mock_llm, temp_db, temp_fm):
        """finalize 在没有 result_text 和 artifacts 时生成默认文本。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="noop",
            intent="unknown",
            status="pending",
        )
        result = await graph.nodes["finalize"].ainvoke(state)
        assert result.status == "completed"
        assert "操作已完成，无额外信息" in result.artifacts[0].get("text", "")

    @pytest.mark.asyncio
    async def test_aggregate_query_drop_sql_raises(self, mock_llm, temp_db, temp_fm):
        """聚合查询生成非 SELECT SQL 时抛出 ValueError。"""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="DROP TABLE project_info")
        )
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="删除项目表",
            intent="query_project",
        )
        with patch("planning_agent.graph._is_aggregate_query", return_value=True):
            with pytest.raises(ValueError):
                await graph.nodes["execute_action"].ainvoke(state)

    @pytest.mark.asyncio
    async def test_aggregate_query_insert_sql_raises(self, mock_llm, temp_db, temp_fm):
        """聚合查询 SQL 包含危险关键字时抛出 ValueError。"""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="SELECT * FROM project_info; INSERT INTO project_info VALUES (1,2,3)"
            )
        )
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="插入数据",
            intent="query_project",
        )
        with patch("planning_agent.graph._is_aggregate_query", return_value=True):
            with pytest.raises(ValueError):
                await graph.nodes["execute_action"].ainvoke(state)

    @pytest.mark.asyncio
    async def test_detail_query_no_matched_project(self, mock_llm, temp_db, temp_fm):
        """明细查询未匹配项目时失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="查一下项目",
            intent="query_project",
            matched_project=None,
        )
        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "未匹配到项目" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_unsupported_intent(self, mock_llm, temp_db, temp_fm):
        """不支持的操作类型导致失败。"""
        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(
            query="unsupported",
            intent="unsupported_intent",
            matched_project=MatchedProject(
                project_name="北京西500千伏输变电工程",
                project_code="PRJ001",
            ),
        )
        result = await graph.nodes["execute_action"].ainvoke(state)
        assert result.status == "failed"
        assert "不支持的操作类型" in (result.result_text or "")

    @pytest.mark.asyncio
    async def test_confirm_project_already_confirmed(self, mock_llm, temp_db, temp_fm):
        """项目已确认时 confirm_project 直接返回。"""
        intent_mock = MagicMock()
        intent_mock.ainvoke = AsyncMock(return_value={"intent": "query_project"})

        filter_mock = MagicMock()
        filter_mock.keywords = "北京西"
        filter_mock.voltage_level = None
        filter_mock.unit_code = None
        filter_mock.min_line_length = None
        filter_mock.max_line_length = None
        filter_mock.min_substation_capacity = None
        filter_mock.max_substation_capacity = None

        mock_llm.with_structured_output = MagicMock(
            side_effect=[
                intent_mock,
                MagicMock(ainvoke=AsyncMock(return_value=filter_mock)),
            ]
        )

        graph = build_planning_graph(mock_llm, temp_db, temp_fm)
        state = PlanningState(query="查一下北京西500千伏项目")
        config = {"configurable": {"thread_id": "test-confirmed-1"}}
        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            await graph.ainvoke(state, config=config)

        # 手动把 checkpoint 中的 project_confirmed 设为 True，
        # 再次恢复时会进入 confirm_project 并命中直接返回分支
        graph.update_state(config, {"project_confirmed": True})

        with patch("planning_agent.graph._is_aggregate_query", return_value=False):
            result = await graph.ainvoke(Command(resume="继续"), config=config)

        assert result.get("project_confirmed") is True
        assert result.get("status") == "completed"


