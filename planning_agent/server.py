"""Planning Agent A2A Server - 基于 DefaultRequestHandler + AgentExecutor。

使用 a2a_base.get_a2a_app 创建 Starlette app，额外挂载 /files/{file_id} 文件下载路由。
"""

from pathlib import Path

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


# ---------- 文件下载路由 ----------


db_instance = ProjectDatabase()
fm_instance = FileManager()


async def download_file(request):
    """文件下载 handler。"""
    file_id = request.path_params["file_id"]
    file_info = db_instance.get_file_by_id(file_id)
    if not file_info:
        return JSONResponse(
            status_code=404, content={"error": f"File not found: {file_id}"}
        )
    file_path = Path(file_info["file_path"])
    if not file_path.exists():
        return JSONResponse(
            status_code=404, content={"error": f"File not found on disk: {file_id}"}
        )
    return FileResponse(
        path=str(file_path),
        filename=file_info.get("file_name", file_path.name),
    )


EXTRA_ROUTES = [
    Route("/files/{file_id}", download_file, methods=["GET"]),
]
