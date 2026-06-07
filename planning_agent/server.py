"""
规划业务 Agent (Planning Agent)
A2A Server - 返回固定的规划测试结果
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
    name="planning-agent",
    description="规划业务 Agent，负责项目规划、任务分解、排期建议等",
    url="http://localhost:8001",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="project-planning",
            name="项目规划",
            description="根据需求进行项目阶段划分、里程碑设定",
            tags=["planning", "project"],
            examples=["帮我规划一个电商网站开发项目", "制定Q3产品迭代计划"],
        ),
        AgentSkill(
            id="task-decomposition",
            name="任务分解",
            description="将大目标拆解为可执行的具体任务",
            tags=["planning", "task"],
            examples=["把上线活动拆解为具体任务", "用户注册功能需要哪些子任务"],
        ),
    ],
)


# ---------- 固定返回值 Handler ----------

FIXED_RESPONSE = """【规划结果 - 固定测试返回】

1. 阶段规划
   - 第一阶段：需求分析与设计（2周）
   - 第二阶段：核心功能开发（4周）
   - 第三阶段：测试与优化（2周）
   - 第四阶段：上线与验收（1周）

2. 关键里程碑
   - M1: 需求评审通过（第2周）
   - M2: 核心模块联调完成（第6周）
   - M3: 测试用例100%通过（第8周）
   - M4: 正式上线（第9周）

3. 风险提示
   - 第三方接口对接需预留缓冲时间
   - 建议每周进行一次进度同步会

(此结果为测试固定返回值，非真实规划)
"""


def handle_task(task: Task) -> Task:
    """处理任务并返回固定规划结果"""
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
            "metadata": {"agent": "planning-agent", "version": "1.0.0"},
        }
    ]
    return task


# ---------- FastAPI App ----------

app = create_a2a_app(agent_card=AGENT_CARD, task_handler=handle_task)
