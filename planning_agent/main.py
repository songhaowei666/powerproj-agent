"""启动入口：规划业务 Agent Server
默认端口 8001
用法：python planning_agent/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a_base import create_server
from planning_agent.executor import PlanningAgentExecutor
from planning_agent.server import AGENT_CARD, EXTRA_ROUTES

if __name__ == "__main__":
    create_server(
        agent_executor=PlanningAgentExecutor(),
        agent_card=AGENT_CARD,
        port=8001,
        extra_routes=EXTRA_ROUTES,
        log_level="debug",
    )
