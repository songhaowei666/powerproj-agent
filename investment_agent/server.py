"""
投资业务 Agent (Investment Agent)
A2A Server - 根据用户输入返回具体电力项目的投资测算结果。
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

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


# ---------- 投资测算系数 ----------
# 单位：万元 / km（线路）、万元 / MVA（变电容量）
# 电压等级越高，单位造价越高；此处为演示用简化系数。
_LINE_UNIT_COST: Dict[str, float] = {
    "10kv": 80.0,
    "35kv": 150.0,
    "220kv": 500.0,
    "330kv": 800.0,
    "500kv": 1200.0,
    "1000kv": 2500.0,
}

_SUBSTATION_UNIT_COST: Dict[str, float] = {
    "10kv": 8.0,
    "35kv": 15.0,
    "220kv": 40.0,
    "330kv": 55.0,
    "500kv": 80.0,
    "1000kv": 150.0,
}

_DEFAULT_LINE_UNIT_COST = 300.0
_DEFAULT_SUBSTATION_UNIT_COST = 30.0


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

    # 按关键词模糊搜索，返回置信度最高的第一条
    keywords = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text).strip()
    if keywords:
        results = db.search_projects(keywords=keywords, limit=1)
        if results:
            return results[0]

    # 未匹配到任何项目时返回第一条种子数据，保证始终给出具体项目示例
    results = db.search_projects(limit=1)
    return results[0] if results else None


def _calculate_investment(project: Dict[str, Any]) -> Dict[str, float]:
    """基于项目电压等级和规模测算线路、变电及总投资。"""
    voltage = (project.get("voltage_level") or "").lower()
    line_length = float(project.get("line_length") or 0.0)
    capacity = float(project.get("substation_capacity") or 0.0)

    line_unit = _LINE_UNIT_COST.get(voltage, _DEFAULT_LINE_UNIT_COST)
    substation_unit = _SUBSTATION_UNIT_COST.get(voltage, _DEFAULT_SUBSTATION_UNIT_COST)

    line_investment = line_length * line_unit
    substation_investment = capacity * substation_unit
    total_investment = line_investment + substation_investment

    return {
        "line_investment": round(line_investment, 2),
        "substation_investment": round(substation_investment, 2),
        "total_investment": round(total_investment, 2),
        "line_unit_cost": line_unit,
        "substation_unit_cost": substation_unit,
    }


def _build_response(project: Dict[str, Any], investment: Dict[str, float]) -> str:
    """组装投资分析返回文本。"""
    return f"""【投资分析结果 - 具体项目】

项目信息
- 项目名称：{project.get('project_name', '未知')}
- 项目编码：{project.get('project_code', '未知')}
- 电压等级：{project.get('voltage_level', '未知')}
- 单位编码：{project.get('unit_code', '未知')}
- 线路长度：{project.get('line_length', 0.0)} km
- 变电容量：{project.get('substation_capacity', 0.0)} MVA

投资测算
- 线路投资：{investment['line_investment']:,.2f} 万元
  （按 {investment['line_unit_cost']} 万元/km 估算）
- 变电投资：{investment['substation_investment']:,.2f} 万元
  （按 {investment['substation_unit_cost']} 万元/MVA 估算）
- 总投资额：{investment['total_investment']:,.2f} 万元

说明
- 以上测算基于项目基础规模与电压等级简化估算，实际投资需结合
  地形、设备选型、材料价格、政策调整等因素进一步校核。
"""


# ---------- Agent Executor ----------

class InvestmentAgentExecutor(AgentExecutor):
    """投资 Agent 执行器：根据用户输入返回具体电力项目投资测算。"""

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

            investment = _calculate_investment(project)
            response_text = _build_response(project, investment)
            final_message = new_text_message(
                text=response_text,
                context_id=task.context_id,
                task_id=task.id,
            )
            await updater.complete(final_message)
        except Exception as exc:
            error_msg = new_text_message(
                text=f"投资分析失败：{type(exc).__name__}: {exc}",
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
    name="investment-agent",
    description="投资业务 Agent，负责电力项目的投资测算、造价分析与投资规模评估",
    version="1.1.0",
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="project-investment-analysis",
            name="项目投资分析",
            description="根据项目编码或项目名称查询具体电力项目，并测算线路、变电及总投资",
            tags=["investment", "power-grid", "project"],
            examples=["分析 PRJ001 的投资规模", "北京西500千伏输变电工程需要投资多少"],
        ),
        AgentSkill(
            id="project-cost-estimation",
            name="项目造价估算",
            description="基于电压等级、线路长度、变电容量等项目参数估算造价",
            tags=["investment", "cost", "estimation"],
            examples=["估算线路长度120km的500kV项目投资", "变电容量1000MVA需要多少投资"],
        ),
    ],
    supported_interfaces=[
        AgentInterface(protocol_binding='JSONRPC', url='http://localhost:8002')
    ],
)
