"""
启动入口：投资业务 Agent Server
默认端口 8002
用法：python investment_agent/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a_base import create_server
from investment_agent.server import InvestmentAgentExecutor, AGENT_CARD

if __name__ == "__main__":
    create_server(
        agent_executor=InvestmentAgentExecutor(),
        agent_card=AGENT_CARD,
        port=8002,
    )
