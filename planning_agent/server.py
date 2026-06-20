"""Planning Agent A2A Server - 基于 DefaultRequestHandler + AgentExecutor。

使用 a2a_base.get_a2a_app 创建 Starlette app，额外挂载文件 HTTP 接口：
- POST /files/upload  multipart/form-data 暂存上传
- GET  /files/{file_id} 下载（暂存或已入库文件）
"""

from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from a2a.types import a2a_pb2

from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.executor import PlanningAgentExecutor


# ---------- Agent Card (protobuf) ----------


def _build_agent_card() -> a2a_pb2.AgentCard:
    """构造 protobuf AgentCard。"""
    card = a2a_pb2.AgentCard()
    card.name = "planning-agent"
    card.description = "规划业务 Agent，负责电力项目信息查询、节点文件上传/下载/删除"
    card.version = "1.1.0"

    # capabilities
    card.capabilities.streaming = False
    card.capabilities.push_notifications = False

    # default modes
    card.default_input_modes.append("text")
    card.default_output_modes.append("text")
    card.default_output_modes.append("file")

    # skills
    skill1 = card.skills.add()
    skill1.id = "project-query"
    skill1.name = "项目信息查询"
    skill1.description = "根据自然语言查询电力项目基本信息及聚合统计"
    skill1.tags.append("planning")
    skill1.tags.append("project")
    skill1.tags.append("query")
    skill1.examples.append("查一下北京西500千伏项目的信息")
    skill1.examples.append("所有项目变电容量的总和是多少")
    skill1.examples.append("线路长度超过100公里的220kv项目有哪些")

    skill2 = card.skills.add()
    skill2.id = "file-management"
    skill2.name = "节点文件管理"
    skill2.description = "按节点编码（001可研设计/002可研评审/003可研批复）上传、下载、删除文件"
    skill2.tags.append("planning")
    skill2.tags.append("file")
    skill2.tags.append("upload")
    skill2.tags.append("download")
    skill2.examples.append("上传可研设计文件到北京西项目")
    skill2.examples.append("下载北京西项目的可研评审文件")
    skill2.examples.append("删除北京西项目可研设计节点的报告.pdf")

    # supported interfaces
    iface = card.supported_interfaces.add()
    iface.protocol_binding = "JSONRPC"
    iface.url = "http://localhost:8001"

    return card


AGENT_CARD = _build_agent_card()


# ---------- 文件 HTTP 接口 ----------


db_instance = ProjectDatabase()
fm_instance = FileManager()


def _request_base_url(request: Request) -> str:
    """构造当前服务的 base URL。"""
    return f"{request.url.scheme}://{request.url.netloc}".rstrip("/")


async def upload_file_http(request: Request):
    """multipart/form-data 暂存上传，返回 file_id 与 download URL。"""
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse(
            status_code=400,
            content={"error": "缺少 form 字段 file"},
        )

    filename = getattr(upload, "filename", None) or "unnamed"
    content = await upload.read()
    if not content:
        return JSONResponse(
            status_code=400,
            content={"error": "上传文件内容为空"},
        )

    file_id = fm_instance.save_staging_file(content, filename)
    base_url = _request_base_url(request)
    return JSONResponse(
        {
            "file_id": file_id,
            "filename": Path(filename).name,
            "url": fm_instance.build_download_url(file_id, base_url),
        }
    )


async def download_file(request: Request):
    """文件下载 handler（支持已入库与暂存文件）。"""
    file_id = request.path_params["file_id"]
    file_info = db_instance.get_file_by_id(file_id)
    download_name = "file"
    file_path: Path | None = None

    if file_info:
        download_name = file_info.get("file_name") or download_name
        file_path = fm_instance.resolve_download_path(
            file_id, file_info.get("file_path")
        )
    else:
        file_path = fm_instance.get_staging_file_path(file_id)
        if file_path is not None:
            download_name = file_path.name

    if file_path is None or not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {file_id}"},
        )

    return FileResponse(
        path=str(file_path),
        filename=download_name,
    )


EXTRA_ROUTES = [
    Route("/files/upload", upload_file_http, methods=["POST"]),
    Route("/files/{file_id}", download_file, methods=["GET"]),
]
