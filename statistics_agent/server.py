"""
统计业务 Agent (Statistics Agent)
A2A Server - 返回固定的统计测试结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from a2a_base import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Task,
    TaskStatus,
    Message,
    create_a2a_app,
)

# ---------- Agent 配置 ----------

AGENT_CARD = AgentCard(
    name="statistics-agent",
    description="统计业务 Agent，负责数据分析、报表生成、指标统计等",
    url="http://localhost:8003",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="data-analysis",
            name="数据分析",
            description="对业务数据进行描述性统计和趋势分析",
            tags=["statistics", "analysis"],
            examples=["分析上月销售数据", "统计用户留存率变化趋势"],
        ),
        AgentSkill(
            id="report-generation",
            name="报表生成",
            description="根据数据自动生成统计报表",
            tags=["statistics", "report"],
            examples=["生成月度运营报表", "输出用户行为统计报表"],
        ),
    ],
)


# ---------- 固定返回值 Handler ----------

FIXED_RESPONSE = """【统计分析结果 - 固定测试返回】

1. 核心指标概览
   - 总用户数：125,430（环比 +5.2%）
   - 日活跃用户(DAU)：18,620（环比 +3.8%）
   - 月活跃用户(MAU)：89,450（环比 +4.1%）
   - 平均会话时长：12分35秒（环比 +8.7%）

2. 业务数据分布
   - 新用户占比：23.5%
   - 回流用户占比：15.2%
   - 留存用户占比：61.3%
   - 流失率：6.8%（环比下降1.2个百分点）

3. 趋势分析
   - 近7日DAU呈稳步上升趋势
   - 周末活跃度较工作日提升约18%
   - 用户平均使用频次：3.2次/天

(此结果为测试固定返回值，非真实统计数据)
"""


def handle_task(task: Task) -> Task:
    """处理任务并返回固定统计结果"""
    task.status = TaskStatus(
        state="completed",
        message=Message(
            role="agent",
            parts=[{"type": "text", "text": FIXED_RESPONSE}],
        ),
    )
    task.artifacts = [
        {
            "type": "text",
            "text": FIXED_RESPONSE,
            "metadata": {"agent": "statistics-agent", "version": "1.0.0"},
        }
    ]
    return task


# ---------- FastAPI App ----------

app = create_a2a_app(agent_card=AGENT_CARD, task_handler=handle_task)
