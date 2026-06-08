"""Pydantic 模型定义 - Planning Agent 状态与项目匹配结果。"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MatchedProject(BaseModel):
    """匹配到的项目信息。"""

    project_name: str = Field(..., description="项目名称")
    project_code: str = Field(..., description="项目编码")
    voltage_level: Optional[str] = Field(default=None, description="电压等级")
    unit_code: Optional[str] = Field(default=None, description="单位编码")
    line_length: Optional[float] = Field(default=None, description="线路长度 (km)")
    substation_capacity: Optional[float] = Field(
        default=None, description="变电容量 (MVA)"
    )
    match_score: float = Field(default=1.0, description="匹配置信度 [0, 1]")


class PlanningState(BaseModel):
    """Planning Agent LangGraph 状态。"""

    query: str = Field(default="", description="当前完整查询文本（含补充信息）")
    intent: str = Field(
        default="unknown",
        description="操作类型：query_project / upload_file / download_file / delete_file / unknown",
    )
    matched_project: Optional[MatchedProject] = Field(
        default=None, description="匹配到的项目"
    )
    project_confirmed: bool = Field(
        default=False, description="用户是否已确认项目"
    )
    node_code: Optional[str] = Field(
        default=None, description="节点编码：001/002/003"
    )
    target_file_id: Optional[str] = Field(
        default=None, description="下载/删除时的目标文件ID"
    )
    target_file_name: Optional[str] = Field(
        default=None, description="删除时通过文件名定位"
    )
    delete_confirmed: bool = Field(
        default=False, description="用户是否已确认删除"
    )
    pending_files: List[Dict[str, Any]] = Field(
        default_factory=list, description="从 A2A message 解析出的待上传文件"
    )
    result_text: Optional[str] = Field(default=None, description="执行结果文本")
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list, description="A2A artifacts"
    )
    status: str = Field(
        default="pending",
        description="整体状态：pending / input_required / completed / failed",
    )
