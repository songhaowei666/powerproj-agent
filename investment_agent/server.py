"""
投资业务 Agent (Investment Agent)
A2A Server - 返回固定的投资测试结果
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard, AgentSkill, AgentInterface, AgentCapabilities,
    TaskState,
)
from a2a.helpers import new_text_message, new_task_from_user_message


# ---------- 固定返回值 ----------

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


# ---------- Agent Executor ----------

class InvestmentAgentExecutor(AgentExecutor):
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
        final_message = new_text_message(
            text=FIXED_RESPONSE,
            context_id=task.context_id,
            task_id=task.id,
        )
        await updater.complete(final_message)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")


# ---------- Agent Card ----------

AGENT_CARD = AgentCard(
    name="investment-agent",
    description="投资业务 Agent，负责投资分析、资产配置建议、风险评估等",
    version="1.0.0",
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=False),
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
    supported_interfaces=[
        AgentInterface(protocol_binding='JSONRPC', url='http://localhost:8002')
    ],
)
