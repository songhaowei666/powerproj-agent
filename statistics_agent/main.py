"""
启动入口：统计业务 Agent Server
默认端口 8003
用法：python statistics_agent/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a_base import create_server
from statistics_agent.server import StatisticsAgentExecutor, AGENT_CARD

if __name__ == "__main__":
    create_server(
        agent_executor=StatisticsAgentExecutor(),
        agent_card=AGENT_CARD,
        port=8003,
    )
