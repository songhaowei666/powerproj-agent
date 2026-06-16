"""
主控 Agent (Main Agent) A2A Server
用户请求入口，负责任务调度与 orchestration。
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    AgentInterface,
)

from a2a_base import create_server
from providers.llm_provider import get_llm
from main_agent.agent_executor import MainAgentExecutor
from main_agent.agent_network import AgentNetwork
from main_agent.registry import DEFAULT_AGENT_URLS


# ---------- Agent 配置 ----------

AGENT_CARD = AgentCard(
    name="main-agent",
    description="主控 Agent，用户请求的统一入口，负责任务调度与 orchestration",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[
        AgentSkill(
            id="task-orchestration",
            name="任务调度",
            description="识别用户意图，按依赖关系分阶段并行调度业务 Agent 执行任务",
            tags=["orchestration", "main"],
            examples=[
                "帮我统计今年的投资收益并做明年规划",
                "分析上月数据并制定下月计划",
            ],
        ),
    ],
    supported_interfaces=[
        AgentInterface(protocol_binding="JSONRPC", url="http://localhost:8000")
    ],
)


# ---------- 全局实例 ----------

_llm = get_llm()
_agent_network = AgentNetwork()
_agent_network.register_from_config(DEFAULT_AGENT_URLS)
_agent_executor = MainAgentExecutor(_llm, _agent_network)


@asynccontextmanager
async def lifespan(app):
    """应用生命周期管理：启动时预发现所有 AgentCard。"""
    await _agent_network.discover()
    yield
    await _agent_network.aclose()


# ---------- 入口 ----------

if __name__ == "__main__":
    create_server(
        agent_executor=_agent_executor,
        agent_card=AGENT_CARD,
        port=8000,
        lifespan=lifespan,
        log_level="info",
    )
