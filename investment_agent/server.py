"""
投资业务 Agent (Investment Agent)
A2A Server - 返回固定的投资测试结果
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
    name="investment-agent",
    description="投资业务 Agent，负责投资分析、资产配置建议、风险评估等",
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="portfolio-analysis",
            name="投资组合分析",
            description="分析现有投资组合的收益与风险",
            tags=["investment", "portfolio"],
            examples=["分析我的当前持仓", "评估这个组合的风险等级"],
        ),
        AgentSkill(
            id="asset-allocation",
            name="资产配置建议",
            description="根据风险偏好提供资产配置方案",
            tags=["investment", "allocation"],
            examples=["稳健型投资者该如何配置资产？", "推荐一个保守的资产配置方案"],
        ),
    ],
)


# ---------- 固定返回值 Handler ----------

FIXED_RESPONSE = """【投资建议 - 固定测试返回】

1. 市场概览
   - A股市场：震荡整理，结构性机会明显
   - 美股市场：科技股表现强劲，关注AI板块
   - 债券市场：利率下行周期，债基配置价值凸显

2. 资产配置建议（稳健型）
   - 权益类资产：40%（沪深300ETF 20%、纳斯达克100 20%）
   - 固收类资产：45%（中短债基金 30%、货币基金 15%）
   - 另类资产：15%（黄金ETF 10%、REITs 5%）

3. 风险提示
   - 当前市场波动率较高，建议分批建仓
   - 单只个股仓位不宜超过总资产的10%
   - 投资有风险，入市需谨慎

(此结果为测试固定返回值，非真实投资建议)
"""


def handle_task(task: Task) -> Task:
    """处理任务并返回固定投资结果"""
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
            "metadata": {"agent": "investment-agent", "version": "1.0.0"},
        }
    ]
    return task


# ---------- FastAPI App ----------

app = create_a2a_app(agent_card=AGENT_CARD, task_handler=handle_task)
