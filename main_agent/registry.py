"""业务 Agent 默认 endpoint 注册表。"""

from typing import List

DEFAULT_AGENT_URLS: List[str] = [
    "http://localhost:8001",  # planning-agent
    "http://localhost:8002",  # investment-agent
    "http://localhost:8003",  # statistics-agent
]
