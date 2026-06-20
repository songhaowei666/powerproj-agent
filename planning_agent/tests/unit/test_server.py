"""server.py / executor.py 单元测试。

使用 Starlette TestClient 对 Planning Agent 进行端到端 JSON-RPC 调用测试，
LLM 使用 Mock 控制返回，确保测试稳定、无外部依赖。
"""

import base64
import os
import shutil
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from a2a_base import get_a2a_app
from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.server import AGENT_CARD, PlanningAgentExecutor, download_file, upload_file_http


# ---------- Mock LLM ----------


class MockLLM:
    """用于单元测试的可控 Mock LLM。"""

    def __init__(self, intent_map=None):
        self._intent_map = intent_map or {}
        self._structured_call_count = 0

    def with_structured_output(self, schema=None, method=None, **kwargs):
        schema_name = getattr(schema, "__name__", str(schema))
        self._structured_call_count += 1

        class StructuredLLM:
            async def ainvoke(inner_self, messages, **kwargs):
                text = ""
                system_text = ""
                if messages:
                    for msg in messages:
                        if isinstance(msg, tuple) and len(msg) > 1:
                            if msg[0] == "system":
                                system_text = str(msg[1])
                            else:
                                text = str(msg[1])
                        elif hasattr(msg, "type"):
                            if getattr(msg, "type", None) == "system":
                                system_text = str(getattr(msg, "content", ""))
                            else:
                                text = str(getattr(msg, "content", ""))
                        elif hasattr(msg, "content"):
                            text = str(msg.content)

                if "dict" in schema_name.lower() and method == "json_mode":
                    # 聚合判断 prompt 包含 "is_aggregate"
                    if "is_aggregate" in system_text:
                        if any(kw in text for kw in ("总和", "平均", "一共", "总计", "统计", "多少", "几个")):
                            return {"is_aggregate": True}
                        return {"is_aggregate": False}

                    # 意图判断
                    for keyword, intent in self._intent_map.items():
                        if keyword in text:
                            return {"intent": intent}
                    return {"intent": "query_project"}

                if "ProjectFilter" in schema_name:
                    return MagicMock(
                        keywords="北京西",
                        voltage_level=None,
                        unit_code=None,
                        min_line_length=None,
                        max_line_length=None,
                        min_substation_capacity=None,
                        max_substation_capacity=None,
                    )

                if "ProjectMatchResult" in schema_name:
                    return MagicMock(
                        project_code="PRJ001",
                        reason="名称匹配",
                        match_score=1.0,
                    )

                return {}

        return StructuredLLM()

    async def ainvoke(self, messages, **kwargs):
        return MagicMock(content="SELECT COUNT(*) AS cnt FROM project_info")


# ---------- Fixtures ----------


@pytest.fixture
def mock_llm():
    return MockLLM(
        intent_map={"上传": "upload_file", "下载": "download_file", "删除": "delete_file"}
    )


@pytest.fixture
def test_client(mock_llm):
    """提供配置好的 TestClient，每个测试独立 DB/FM/Executor/App。"""
    db_dir = tempfile.mkdtemp()
    fm_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "test.db")
    db = ProjectDatabase(db_path=db_path)
    fm = FileManager(base_dir=fm_dir)
    executor = PlanningAgentExecutor(llm=mock_llm, db=db, fm=fm)

    # 测试专用文件 HTTP 路由（与 server.EXTRA_ROUTES 一致）
    async def _upload_file(request):
        with patch("planning_agent.server.fm_instance", fm):
            return await upload_file_http(request)

    async def _download_file(request):
        with patch("planning_agent.server.db_instance", db), patch(
            "planning_agent.server.fm_instance", fm
        ):
            return await download_file(request)

    extra_routes = [
        Route("/files/upload", _upload_file, methods=["POST"]),
        Route("/files/{file_id}", _download_file, methods=["GET"]),
    ]

    app = get_a2a_app(executor, AGENT_CARD, extra_routes=extra_routes)

    with TestClient(app) as client:
        yield client

    # 清理临时文件
    os.unlink(db_path)
    shutil.rmtree(db_dir, ignore_errors=True)
    shutil.rmtree(fm_dir, ignore_errors=True)


# ---------- Helpers ----------


def _rpc(method: str, params: dict, req_id: int = 1) -> dict:
    # 自动为 message 注入 messageId，满足 protobuf 校验
    msg = params.get("message")
    if msg and not msg.get("messageId"):
        msg["messageId"] = str(uuid.uuid4())
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}


A2A_HEADERS = {"A2A-Version": "1.0"}


def _get_task_from_result(resp_json: dict) -> dict:
    """从 SendMessage 响应中提取 task 对象。"""
    result = resp_json.get("result", {})
    return result.get("task", result)


# ---------- Tests ----------


class TestAgentCard:
    """Agent Card 路由测试。"""

    def test_get_agent_card(self, test_client):
        """GET /.well-known/agent-card.json 返回正确。"""
        resp = test_client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "planning-agent"
        assert len(data.get("skills", [])) >= 2


class TestTasksSend:
    """SendMessage 端到端测试。"""

    def test_tasks_send_new_project_query(self, test_client):
        """新项目查询返回 input-required，包含匹配的项目信息。"""
        payload = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "查一下北京西500千伏项目"}],
                }
            },
        )
        resp = test_client.post("/", json=payload, headers=A2A_HEADERS)
        assert resp.status_code == 200
        task = _get_task_from_result(resp.json())
        assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        text = task["status"]["message"]["parts"][0]["text"]
        assert "北京西500千伏输变电工程" in text
        assert "PRJ001" in text

    def test_tasks_send_confirm_project(self, test_client):
        """确认项目后返回 completed，包含项目详情。"""
        # 第一轮：查询（不指定 taskId，由 SDK 生成）
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "查一下北京西500千伏项目"}],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        task_id = task1["id"]

        # 第二轮：确认
        payload2 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": task_id,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp2 = test_client.post("/", json=payload2, headers=A2A_HEADERS)
        assert resp2.status_code == 200
        task = _get_task_from_result(resp2.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        artifacts = task.get("artifacts", [])
        assert len(artifacts) > 0
        assert "PRJ001" in artifacts[0]["parts"][0]["text"]

    def test_tasks_send_aggregate_query(self, test_client):
        """聚合查询直接 completed，跳过项目确认。"""
        payload = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "所有项目变电容量的总和"}],
                }
            },
        )
        resp = test_client.post("/", json=payload, headers=A2A_HEADERS)
        assert resp.status_code == 200
        task = _get_task_from_result(resp.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        artifacts = task.get("artifacts", [])
        assert len(artifacts) > 0

    def test_tasks_send_upload_file(self, test_client):
        """上传文件：先确认项目，再完成上传。"""
        file_content = base64.b64encode(b"test file content").decode()
        # 第一轮：发送文件 + 说明（不指定 taskId）
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": "上传可研设计文件到北京西项目"},
                        {"raw": file_content, "filename": "design.pdf"},
                    ],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        task_id = task1["id"]

        # 第二轮：确认项目
        payload2 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": task_id,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp2 = test_client.post("/", json=payload2, headers=A2A_HEADERS)
        assert resp2.status_code == 200
        task = _get_task_from_result(resp2.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert "成功上传" in task["artifacts"][0]["parts"][0]["text"]

    def test_tasks_send_download_file(self, test_client):
        """下载文件：先上传文件，再查询下载。"""
        file_content = base64.b64encode(b"report content").decode()
        # 第一轮：上传（不指定 taskId）
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": "上传可研设计文件到北京西项目"},
                        {"raw": file_content, "filename": "report.pdf"},
                    ],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        task_id = task1["id"]

        # 第二轮：确认上传
        payload2 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": task_id,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp2 = test_client.post("/", json=payload2, headers=A2A_HEADERS)
        assert _get_task_from_result(resp2.json())["status"]["state"] == "TASK_STATE_COMPLETED"

        # 第三轮：请求下载（新 task）
        payload3 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "下载北京西项目的可研设计文件"}],
                }
            },
        )
        resp3 = test_client.post("/", json=payload3, headers=A2A_HEADERS)
        task3 = _get_task_from_result(resp3.json())
        assert task3["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        task_id2 = task3["id"]

        # 第四轮：确认项目
        payload4 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": task_id2,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp4 = test_client.post("/", json=payload4, headers=A2A_HEADERS)
        assert resp4.status_code == 200
        task = _get_task_from_result(resp4.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        # 检查是否有 file 类型的 artifact（包含 url）
        file_artifacts = [
            a
            for a in task.get("artifacts", [])
            if any(p.get("url") for p in a.get("parts", []))
        ]
        assert len(file_artifacts) > 0


class TestTasksGetCancel:
    """GetTask / CancelTask 测试。"""

    def test_tasks_get(self, test_client):
        """根据 task_id 查询已存在的任务。"""
        # 先创建任务（不指定 taskId）
        payload = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "所有项目变电容量的总和"}],
                }
            },
        )
        resp = test_client.post("/", json=payload, headers=A2A_HEADERS)
        task_id = _get_task_from_result(resp.json())["id"]

        # 查询任务
        get_payload = _rpc("GetTask", {"id": task_id})
        resp = test_client.post("/", json=get_payload, headers=A2A_HEADERS)
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["id"] == task_id
        assert result["status"]["state"] in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_INPUT_REQUIRED",
            "TASK_STATE_FAILED",
        )

    def test_tasks_cancel(self, test_client):
        """取消任务后状态变为 canceled。"""
        # 先创建任务（不指定 taskId），使用会中断的查询以保留任务在 non-terminal 状态
        payload = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "查一下北京西500千伏项目"}],
                }
            },
        )
        resp = test_client.post("/", json=payload, headers=A2A_HEADERS)
        task_id = _get_task_from_result(resp.json())["id"]

        # 取消任务
        cancel_payload = _rpc("CancelTask", {"id": task_id})
        resp = test_client.post("/", json=cancel_payload, headers=A2A_HEADERS)
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["status"]["state"] == "TASK_STATE_CANCELED"


class TestFileUploadHttp:
    """HTTP 暂存上传接口测试。"""

    def test_upload_and_download_staging(self, test_client):
        """POST /files/upload 后可通过 GET /files/{id} 下载。"""
        resp = test_client.post(
            "/files/upload",
            files={"file": ("report.pdf", b"staging-content", "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_id"]
        assert data["url"].endswith(f"/files/{data['file_id']}")

        download = test_client.get(f"/files/{data['file_id']}")
        assert download.status_code == 200
        assert download.content == b"staging-content"


class TestFileDownload:
    """文件下载路由测试。"""

    def test_download_file(self, test_client):
        """GET /files/{file_id} 返回正确的文件。

        先上传文件，再执行下载查询，从下载响应的 artifacts 中提取 file_id。
        """
        file_content = base64.b64encode(b"report content").decode()
        # 第一轮：上传文件（不指定 taskId）
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": "上传可研设计文件到北京西项目"},
                        {"raw": file_content, "filename": "report.pdf"},
                    ],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        upload_task_id = task1["id"]

        # 第二轮：确认上传
        payload2 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": upload_task_id,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp2 = test_client.post("/", json=payload2, headers=A2A_HEADERS)
        assert _get_task_from_result(resp2.json())["status"]["state"] == "TASK_STATE_COMPLETED"

        # 第三轮：请求下载（新 task）
        payload3 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "下载北京西项目的可研设计文件"}],
                }
            },
        )
        resp3 = test_client.post("/", json=payload3, headers=A2A_HEADERS)
        task3 = _get_task_from_result(resp3.json())
        assert task3["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        download_task_id = task3["id"]

        # 第四轮：确认项目
        payload4 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "taskId": download_task_id,
                    "parts": [{"text": "是的"}],
                }
            },
        )
        resp4 = test_client.post("/", json=payload4, headers=A2A_HEADERS)
        task = _get_task_from_result(resp4.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"

        # 从下载响应的 artifacts 中提取 file_id
        file_artifact = None
        for a in task.get("artifacts", []):
            for p in a.get("parts", []):
                url = p.get("url", "")
                if "/files/" in url:
                    file_id = url.split("/files/")[-1]
                    file_artifact = file_id
                    break
            if file_artifact:
                break

        assert file_artifact is not None, "未找到上传文件的 file_id"

        resp = test_client.get(f"/files/{file_artifact}")
        assert resp.status_code == 200
        assert resp.content == b"report content"
        assert "report.pdf" in resp.headers.get("content-disposition", "")

    def test_download_file_not_found(self, test_client):
        """GET /files/not-exist-uuid 返回 404。"""
        resp = test_client.get("/files/not-exist-uuid")
        assert resp.status_code == 404


class TestServerDownloadFile:
    """直接测试 planning_agent.server.download_file handler。"""

    @pytest.mark.asyncio
    async def test_download_file_not_in_db(self):
        """数据库无记录时返回 404。"""
        request = MagicMock()
        request.path_params = {"file_id": "not-exist"}

        with patch("planning_agent.server.db_instance.get_file_by_id", return_value=None):
            with patch(
                "planning_agent.server.fm_instance.get_staging_file_path",
                return_value=None,
            ):
                resp = await download_file(request)
                assert resp.status_code == 404
                assert "not-exist" in resp.body.decode()

    @pytest.mark.asyncio
    async def test_download_file_missing_on_disk(self):
        """数据库有记录但磁盘文件不存在时返回 404。"""
        request = MagicMock()
        request.path_params = {"file_id": "missing-file"}

        file_info = {
            "file_id": "missing-file",
            "file_name": "gone.pdf",
            "file_path": "/tmp/planning_agent_tests/not_exist/gone.pdf",
        }

        with patch("planning_agent.server.db_instance.get_file_by_id", return_value=file_info):
            resp = await download_file(request)
            assert resp.status_code == 404
            assert "File not found" in resp.body.decode()

    @pytest.mark.asyncio
    async def test_download_file_success(self):
        """数据库有记录且磁盘文件存在时返回 FileResponse。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "report.pdf")
            with open(file_path, "wb") as f:
                f.write(b"report content")

            request = MagicMock()
            request.path_params = {"file_id": "real-file"}

            file_info = {
                "file_id": "real-file",
                "file_name": "report.pdf",
                "file_path": file_path,
            }

            with patch(
                "planning_agent.server.db_instance.get_file_by_id", return_value=file_info
            ):
                resp = await download_file(request)
                assert isinstance(resp, FileResponse)
                assert resp.path == file_path
                assert resp.filename == "report.pdf"
