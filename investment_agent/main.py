"""
启动入口：投资业务 Agent Server
默认端口 8002
用法：python investment_agent/main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
