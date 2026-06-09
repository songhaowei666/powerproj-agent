"""任务规划示例脚本。

演示如何基于 A2A AgentCard 中的技能描述，将复杂任务分解为可执行的子任务序列。
该脚本模拟从 A2A 服务发现获取 AgentCard，提取可用能力，并调用 LLM 进行任务规划。
"""

import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保能导入 providers 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import asyncio
import json
from typing import List, Sequence

from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from providers.llm_provider import get_llm
from examples.agents.task_planning_and_dispatch.prompts import TASK_PLANNING_PROMPT


class SubTask(BaseModel):
    """子任务定义。"""

    id: str = Field(..., description="子任务编号")
    name: str = Field(..., description="子任务名称")
    description: str = Field(..., description="详细描述")
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的子任务 ID 列表"
    )
    expected_output: str = Field(..., description="预期输出描述")
    required_capability: str = Field(..., description="所需 Agent 能力类型")


class TaskPlan(BaseModel):
    """任务规划结果。"""

    task_goal: str = Field(..., description="原始任务目标概述")
    subtasks: List[SubTask] = Field(..., description="子任务列表")
    execution_order: List[str] = Field(
        ..., description="按执行顺序排列的子任务 ID"
    )


class TaskPlanningAgent:
    """任务规划 Agent。

    基于 A2A AgentCard 中的技能描述，将复杂任务分解为可执行的子任务序列。
    支持 protobuf 风格与 dataclass 风格的 AgentCard 对象。
    """

    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm or get_llm()

    @staticmethod
    def _extract_capabilities_from_cards(agent_cards: Sequence) -> str:
        """从 AgentCard 列表中提取并格式化能力描述。

        Args:
            agent_cards: A2A AgentCard 对象列表。

        Returns:
            格式化后的能力描述文本，用于填充 prompt 占位符。
        """
        capabilities: List[str] = []
        for card in agent_cards:
            card_name = getattr(card, "name", "未知 Agent")
            skills = getattr(card, "skills", [])
            for skill in skills:
                skill_id = getattr(skill, "id", "")
                skill_name = getattr(skill, "name", "")
                skill_desc = getattr(skill, "description", "")
                tags = getattr(skill, "tags", [])
                tags_str = ", ".join(str(t) for t in tags) if tags else "无"
                examples = getattr(skill, "examples", [])
                examples_str = ""
                if examples:
                    examples_str = (
                        f"\n  示例：{'; '.join(str(e) for e in examples)}"
                    )

                capabilities.append(
                    f"- `{skill_id}` — {skill_name}\n"
                    f"  所属Agent：{card_name}\n"
                    f"  描述：{skill_desc}\n"
                    f"  标签：{tags_str}{examples_str}"
                )

        return (
            "\n".join(capabilities)
            if capabilities
            else "（暂无可用 Agent 能力）"
        )

    async def plan(self, task_description: str, agent_cards: Sequence) -> TaskPlan:
        """将复杂任务分解为子任务序列。

        Args:
            task_description: 用户输入的复杂任务描述。
            agent_cards: 可用 Agent 的 AgentCard 列表。

        Returns:
            TaskPlan: 分解后的任务计划。

        Raises:
            ValueError: 当 LLM 返回的结果无法解析为有效任务计划时。
        """
        capabilities_text = self._extract_capabilities_from_cards(agent_cards)
        system_prompt = TASK_PLANNING_PROMPT.format(
            agent_capabilities=capabilities_text,
            user_query=task_description,
        )

        structured_llm = self.llm.with_structured_output(
            schema=TaskPlan,
            method="json_mode",
        )

        result = await structured_llm.ainvoke(
            [
                ("system", system_prompt),
                ("human", "请按上述要求输出任务分解结果。"),
            ]
        )

        if not isinstance(result, TaskPlan):
            raise ValueError(
                f"LLM 返回了非预期的结果类型: {type(result)}"
            )

        return result


# ---------- 示例数据 ----------

def _build_sample_agent_cards() -> List:
    """构造示例 AgentCard 列表，模拟从 A2A 服务发现获取。"""
    try:
        from a2a.types import AgentCard, AgentSkill, AgentCapabilities

        data_skill = AgentSkill(
            id="data-query",
            name="数据查询",
            description="从数据库或 API 检索结构化数据，支持 SQL 聚合统计",
            tags=["data", "query", "sql"],
            examples=["查询所有项目的变电容量总和", "获取北京西 500kV 项目的基本信息"],
        )

        doc_skill = AgentSkill(
            id="doc-analysis",
            name="文档分析",
            description="解析 PDF、Word、Excel 等文档内容，提取关键信息",
            tags=["document", "analysis", "pdf", "excel"],
            examples=["解析可研设计报告中的线路长度", "提取评审意见中的问题清单"],
        )

        report_skill = AgentSkill(
            id="report-generation",
            name="报告生成",
            description="根据数据和分析结果生成结构化报告或演示文稿",
            tags=["report", "generation", "markdown", "ppt"],
            examples=["生成本月项目进度报告", "制作投资计划汇总 PPT"],
        )

        validation_skill = AgentSkill(
            id="result-validation",
            name="结果校验",
            description="检查数据和输出结果的正确性、完整性和一致性",
            tags=["validation", "check", "quality"],
            examples=["校验项目编码格式是否符合规范", "检查报告数据与原始数据是否一致"],
        )

        card1 = AgentCard(
            name="数据分析 Agent",
            description="负责数据检索与统计分析",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=False),
            skills=[data_skill],
        )

        card2 = AgentCard(
            name="文档处理 Agent",
            description="负责文档解析与信息提取",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=False),
            skills=[doc_skill],
        )

        card3 = AgentCard(
            name="报告生成 Agent",
            description="负责报告撰写与格式输出",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=False),
            skills=[report_skill],
        )

        card4 = AgentCard(
            name="质量校验 Agent",
            description="负责结果审核与质量把关",
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=False),
            skills=[validation_skill],
        )

        return [card1, card2, card3, card4]

    except ImportError:
        # 降级：使用纯字典模拟 AgentCard
        return [
            {
                "name": "数据分析 Agent",
                "skills": [
                    {
                        "id": "data-query",
                        "name": "数据查询",
                        "description": "从数据库或 API 检索结构化数据，支持 SQL 聚合统计",
                        "tags": ["data", "query", "sql"],
                        "examples": ["查询所有项目的变电容量总和"],
                    }
                ],
            },
            {
                "name": "文档处理 Agent",
                "skills": [
                    {
                        "id": "doc-analysis",
                        "name": "文档分析",
                        "description": "解析 PDF、Word、Excel 等文档内容，提取关键信息",
                        "tags": ["document", "analysis", "pdf"],
                        "examples": ["解析可研设计报告中的线路长度"],
                    }
                ],
            },
            {
                "name": "报告生成 Agent",
                "skills": [
                    {
                        "id": "report-generation",
                        "name": "报告生成",
                        "description": "根据数据和分析结果生成结构化报告或演示文稿",
                        "tags": ["report", "generation", "markdown"],
                        "examples": ["生成本月项目进度报告"],
                    }
                ],
            },
            {
                "name": "质量校验 Agent",
                "skills": [
                    {
                        "id": "result-validation",
                        "name": "结果校验",
                        "description": "检查数据和输出结果的正确性、完整性和一致性",
                        "tags": ["validation", "check", "quality"],
                        "examples": ["校验项目编码格式是否符合规范"],
                    }
                ],
            },
        ]


# ---------- 入口 ----------


async def main() -> None:
    """命令行入口：演示任务规划完整流程。"""
    agent = TaskPlanningAgent()
    agent_cards = _build_sample_agent_cards()

    task_description = (
        "分析本季度所有 220kV 以上电力项目的投资完成情况，"
        # "提取各项目的可研批复文件中的概算金额，"
        # "生成一份投资完成率分析报告，"
        # "并校验报告中的数据与原始数据库记录的一致性。"
    )

    print("=" * 60)
    print("任务描述：")
    print(task_description)
    print("=" * 60)
    print("\n可用 Agent 能力：")
    print(agent._extract_capabilities_from_cards(agent_cards))
    print("\n" + "=" * 60)
    print("开始规划...\n")

    plan = await agent.plan(task_description, agent_cards)

    print("规划结果：")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
