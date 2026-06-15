"""
统计业务 Agent (Statistics Agent)
A2A Server - 根据用户输入返回具体电力项目的统计指标。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard, AgentSkill, AgentInterface, AgentCapabilities,
    TaskState,
)
from a2a.helpers import new_text_message, new_task_from_user_message, get_message_text

from a2a_message_parser import parse_message_parts
from planning_agent.database import ProjectDatabase


# ---------- 统计维度系数 ----------
# 用于生成附加业务统计指标，与 planning_agent 共享同一套项目数据。

_VOLTAGE_WEIGHTS: Dict[str, float] = {
    "10kv": 1.0,
    "35kv": 2.0,
    "220kv": 4.0,
    "330kv": 5.0,
    "500kv": 7.0,
    "1000kv": 10.0,
}

_DEFAULT_VOLTAGE_WEIGHT = 3.0


def _resolve_db_path() -> str:
    """返回与 planning_agent 共享的 SQLite 数据库绝对路径。"""
    project_root = Path(__file__).parent.parent
    return str(project_root / "planning_agent" / "planning.db")


def _extract_project_code(text: str) -> Optional[str]:
    """从用户文本中提取项目编码（PRJ + 三位数字）。"""
    if not text:
        return None
    match = re.search(r"PRJ\d{3}", text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _find_project(db: ProjectDatabase, text: str) -> Optional[Dict[str, Any]]:
    """根据文本中的项目编码或关键词匹配一个具体项目。"""
    code = _extract_project_code(text)
    if code:
        return db.get_project_by_code(code)

    keywords = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text).strip()
    if keywords:
        results = db.search_projects(keywords=keywords, limit=1)
        if results:
            return results[0]

    # 未匹配时返回第一条数据，保证始终给出具体项目示例
    results = db.search_projects(limit=1)
    return results[0] if results else None


def _calculate_statistics(project: Dict[str, Any], all_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算具体项目的统计指标及在全部项目中的相对位置。"""
    voltage = (project.get("voltage_level") or "").lower()
    line_length = float(project.get("line_length") or 0.0)
    capacity = float(project.get("substation_capacity") or 0.0)

    total_line_length = sum(float(p.get("line_length") or 0.0) for p in all_projects)
    total_capacity = sum(float(p.get("substation_capacity") or 0.0) for p in all_projects)

    line_length_ratio = (line_length / total_line_length * 100) if total_line_length else 0.0
    capacity_ratio = (capacity / total_capacity * 100) if total_capacity else 0.0

    sorted_by_line = sorted(all_projects, key=lambda p: float(p.get("line_length") or 0.0), reverse=True)
    sorted_by_capacity = sorted(all_projects, key=lambda p: float(p.get("substation_capacity") or 0.0), reverse=True)

    line_rank = next((i + 1 for i, p in enumerate(sorted_by_line) if p.get("project_code") == project.get("project_code")), len(all_projects))
    capacity_rank = next((i + 1 for i, p in enumerate(sorted_by_capacity) if p.get("project_code") == project.get("project_code")), len(all_projects))

    voltage_weight = _VOLTAGE_WEIGHTS.get(voltage, _DEFAULT_VOLTAGE_WEIGHT)
    scale_score = round((line_length / 10.0 + capacity / 100.0) * voltage_weight, 2)

    return {
        "total_projects": len(all_projects),
        "line_length_ratio": round(line_length_ratio, 2),
        "capacity_ratio": round(capacity_ratio, 2),
        "line_rank": line_rank,
        "capacity_rank": capacity_rank,
        "scale_score": scale_score,
    }


def _build_response(project: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """组装统计分析返回文本。"""
    return f"""【统计分析结果 - 具体项目】

项目信息
- 项目名称：{project.get('project_name', '未知')}
- 项目编码：{project.get('project_code', '未知')}
- 电压等级：{project.get('voltage_level', '未知')}
- 单位编码：{project.get('unit_code', '未知')}

规模统计
- 线路长度：{project.get('line_length', 0.0)} km
- 变电容量：{project.get('substation_capacity', 0.0)} MVA
- 规模评分：{stats['scale_score']} 分（基于电压等级、线路长度与变电容量综合计算）

相对位置（共 {stats['total_projects']} 个项目）
- 线路长度排名：第 {stats['line_rank']} 位
  （占全部项目线路总长度的 {stats['line_length_ratio']}%）
- 变电容量排名：第 {stats['capacity_rank']} 位
  （占全部项目变电总容量的 {stats['capacity_ratio']}%）

说明
- 规模评分与排名仅用于横向对比参考，实际项目评估需结合
  投资、工期、地理条件、社会效益等多维因素综合分析。
"""


# ---------- Agent Executor ----------

class StatisticsAgentExecutor(AgentExecutor):
    """统计 Agent 执行器：根据用户输入返回具体电力项目统计指标。"""

    def __init__(self, db: Optional[ProjectDatabase] = None):
        self._db = db or ProjectDatabase(db_path=_resolve_db_path())

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            parsed = parse_message_parts(context.message)
            text = parsed.task_query or (
                get_message_text(context.message) if context.message else ""
            )
            project = _find_project(self._db, text)

            if not project:
                error_msg = new_text_message(
                    text="未找到任何项目信息，请确认项目编码或项目名称后重试。",
                    context_id=task.context_id,
                    task_id=task.id,
                )
                await updater.failed(error_msg)
                return

            all_projects = self._db.search_projects(limit=1000)
            stats = _calculate_statistics(project, all_projects)
            response_text = _build_response(project, stats)
            final_message = new_text_message(
                text=response_text,
                context_id=task.context_id,
                task_id=task.id,
            )
            await updater.complete(final_message)
        except Exception as exc:
            error_msg = new_text_message(
                text=f"统计分析失败：{type(exc).__name__}: {exc}",
                context_id=task.context_id,
                task_id=task.id,
            )
            await updater.failed(error_msg)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")


# ---------- Agent Card ----------

AGENT_CARD = AgentCard(
    name="statistics-agent",
    description="统计业务 Agent，负责电力项目的规模统计、排名分析与指标对比",
    version="1.1.0",
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="project-statistics",
            name="项目统计指标",
            description="根据项目编码或项目名称查询具体电力项目，输出线路长度、变电容量及相对排名",
            tags=["statistics", "power-grid", "project"],
            examples=["统计 PRJ001 的指标", "北京西500千伏输变电工程的规模排名如何"],
        ),
        AgentSkill(
            id="project-scale-comparison",
            name="项目规模对比",
            description="计算具体项目在所有项目中的线路长度、变电容量的占比与排名",
            tags=["statistics", "comparison", "ranking"],
            examples=["哪个项目变电容量最大", "PRJ003 的线路长度占比是多少"],
        ),
    ],
    supported_interfaces=[
        AgentInterface(protocol_binding='JSONRPC', url='http://localhost:8003')
    ],
)
