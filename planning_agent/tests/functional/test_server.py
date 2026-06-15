"""server.py 功能测试。

使用真实 LLM 与开发环境数据（planning_agent/planning.db、upload_files）
对 Planning Agent 进行端到端 JSON-RPC 调用测试。
这些测试默认跳过，需设置环境变量 RUN_FUNCTIONAL_TESTS=1 才会执行，
因为它们会调用真实的 LLM API 并产生费用，且会读写开发环境数据库与文件。
"""

import base64
import os
import uuid

import pytest
from starlette.testclient import TestClient

from a2a_base import get_a2a_app
from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.server import AGENT_CARD, PlanningAgentExecutor, download_file
from planning_agent.server import EXTRA_ROUTES


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_FUNCTIONAL_TESTS") != "1",
    reason="功能测试默认跳过，设置 RUN_FUNCTIONAL_TESTS=1 后启用（会产生 LLM 费用）",
)


@pytest.fixture
def test_client():
    """提供使用真实 LLM 与开发环境数据库/文件目录的 TestClient。"""
    # 与 server.py 一致，复用 planning_agent/planning.db 与 upload_files
    db = ProjectDatabase()
    fm = FileManager()

    # 使用 providers 中的真实 LLM
    from providers.llm_provider import get_llm

    executor = PlanningAgentExecutor(llm=get_llm(), db=db, fm=fm)
    app = get_a2a_app(executor, AGENT_CARD, extra_routes=EXTRA_ROUTES)

    with TestClient(app) as client:
        yield client


# ---------- Helpers ----------


def _rpc(method: str, params: dict, req_id: int = 1) -> dict:
    msg = params.get("message")
    if msg and not msg.get("messageId"):
        msg["messageId"] = str(uuid.uuid4())
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": req_id}


A2A_HEADERS = {"A2A-Version": "1.0"}


def _get_task_from_result(resp_json: dict) -> dict:
    result = resp_json.get("result", {})
    return result.get("task", result)


# ---------- Tests ----------


class TestAgentCard:
    """Agent Card 路由功能测试。"""

    def test_get_agent_card(self, test_client):
        """GET /.well-known/agent-card.json 返回正确。"""
        resp = test_client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "planning-agent"
        assert len(data.get("skills", [])) >= 2


class TestAggregateQuery:
    """聚合查询功能测试。"""

    def test_aggregate_query(self, test_client):
        """聚合查询直接返回 completed。"""
        payload = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "所有项目变电容量的总和是多少"}],
                }
            },
        )
        resp = test_client.post("/", json=payload, headers=A2A_HEADERS)
        assert resp.status_code == 200
        task = _get_task_from_result(resp.json())
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        artifacts = task.get("artifacts", [])
        assert len(artifacts) > 0
        text = artifacts[0]["parts"][0].get("text", "")
        # 真实 LLM 可能返回中文数字或描述，只要包含结果即可
        assert "变电" in text or "容量" in text or "总和" in text


class TestProjectQueryAndConfirm:
    """项目查询 + 确认功能测试。"""

    def test_project_query_and_confirm(self, test_client):
        """查询项目后确认，返回 completed 详情。"""
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "查一下北京西500千伏输变电工程"}],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        task_id = task1["id"]

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
        text = artifacts[0]["parts"][0].get("text", "")
        assert "PRJ001" in text or "北京西" in text


class TestFileUploadAndDownload:
    """文件上传下载功能测试。"""

    def test_upload_and_download_file(self, test_client):
        """上传文件后通过下载路由获取。"""
        file_content = base64.b64encode(b"functional test content").decode()
        # 上传
        payload1 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [
                        {"text": "上传可研设计文件到北京西500千伏输变电工程"},
                        {"raw": file_content, "filename": "func.pdf"},
                    ],
                }
            },
        )
        resp1 = test_client.post("/", json=payload1, headers=A2A_HEADERS)
        task1 = _get_task_from_result(resp1.json())
        assert task1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        upload_task_id = task1["id"]

        # 确认上传
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
        task2 = _get_task_from_result(resp2.json())
        assert task2["status"]["state"] == "TASK_STATE_COMPLETED"

        # 查询下载
        payload3 = _rpc(
            "SendMessage",
            {
                "message": {
                    "role": "ROLE_USER",
                    "parts": [{"text": "下载北京西500千伏输变电工程可研设计文件"}],
                }
            },
        )
        resp3 = test_client.post("/", json=payload3, headers=A2A_HEADERS)
        task3 = _get_task_from_result(resp3.json())
        assert task3["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
        download_task_id = task3["id"]

        # 确认下载
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
        task4 = _get_task_from_result(resp4.json())
        assert task4["status"]["state"] == "TASK_STATE_COMPLETED"

        # 提取 file_id 并访问下载路由
        file_id = None
        for artifact in task4.get("artifacts", []):
            for part in artifact.get("parts", []):
                url = part.get("url", "")
                if "/files/" in url:
                    file_id = url.split("/files/")[-1]
                    break
            if file_id:
                break

        assert file_id is not None
        resp = test_client.get(f"/files/{file_id}")
        assert resp.status_code == 200
        assert resp.content == b"functional test content"
