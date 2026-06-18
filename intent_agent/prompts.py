"""System prompt 模板 + AgentCard 能力提取 + 少样本拼接逻辑。"""

import json
from typing import List, Dict, Sequence

from intent_agent.models import IntentResult


_TASK_PLANNING_RULES = """\
## 任务分解原则
1. 原子性：每个子任务应足够具体，可由单个业务 Agent 独立完成
2. 完整性：所有子任务的组合必须能够完成原始任务目标
3. 有序性：明确标注子任务之间的依赖关系和执行顺序
4. 可验证性：每个子任务应有明确的完成标准
5. Agent 匹配：每个子任务的 required_agent 必须从上方可用业务 Agent 列表中选取
"""

_OUTPUT_RULES = """\
## 输出要求
- 首先判断用户 query 是否为电网业务相关查询，填写 is_business_query：
  - true：涉及项目查询、统计分析、规划、投资评估等业务意图，或业务意图不明确但可能属于上述范畴
  - false：问候、闲聊、感谢、与电网业务无关的内容
- 当 is_business_query 为 false 时：
  - subtasks 必须为 []，execution_order 必须为 []
  - clarification_prompt 必须为 null
- 当 is_business_query 为 true 但无法拆出具体子任务时：
  - subtasks 为 []，execution_order 为 []
  - clarification_prompt 必须填写：1~3 句面向用户的澄清问句，结合用户原话与可用 Agent 举例，引导其说明具体需求
- 当 is_business_query 为 true 且已拆出子任务，但 Agent 匹配或信息仍不足时：
  - 仍输出 subtasks，并在 clarification_prompt 中给出针对性澄清问句（可为 null，若信息已充分）
- 仔细分析用户 query，识别其中涉及的一个或多个业务意图
- 将每个意图拆分为可由单个业务 Agent 完成的子任务
- 每个子任务必须包含：id, name, description, dependencies, expected_output, required_agent
- id 从 task_1 开始顺序编号，如 task_1, task_2, task_3
- dependencies 使用前置子任务的 id 列表表示依赖关系，无依赖则为空列表 []
- required_agent 必须匹配上方可用业务 Agent 列表中的某个 Agent 名称（AgentCard.name）
- 如果多个子任务之间存在先后依赖关系，请在 dependencies 中正确声明
- description 用一句话概括该子任务的具体内容
- expected_output 说明该子任务完成后应产生的具体结果
- clarification_prompt 必须是直接对用户说的话，不要使用「系统」「模型」等内部用语

## 输出格式

严格遵循以下 JSON Schema：
{schema}
"""


def _format_few_shots(few_shots: List[Dict]) -> str:
    """将少样本示例格式化为 prompt 文本。"""
    if not few_shots:
        return ""

    lines = ["## 少样本示例\n"]
    for idx, example in enumerate(few_shots, 1):
        lines.append(f"### 示例 {idx}")
        lines.append(f"用户 query：{example.get('query', '')}")
        lines.append("输出：")
        lines.append(json.dumps(example.get("tasks", []), ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)


def _get_card_field(card: object, field: str, default: object = "") -> object:
    """从 AgentCard 对象或序列化 dict 中读取字段。"""
    if isinstance(card, dict):
        return card.get(field, default)
    return getattr(card, field, default)


def _extract_agents(agent_cards: Sequence) -> str:
    """从 AgentCard 列表中提取并格式化业务 Agent 描述。

    Args:
        agent_cards: A2A AgentCard 对象列表或序列化 dict 列表

    Returns:
        格式化后的业务 Agent 描述文本
    """
    agents: List[str] = []
    for card in agent_cards:
        card_name = _get_card_field(card, "name", "未知 Agent")
        card_desc = _get_card_field(card, "description", "")
        skills = _get_card_field(card, "skills", [])
        skill_lines: List[str] = []
        for skill in skills:
            skill_name = _get_card_field(skill, "name", "")
            skill_desc = _get_card_field(skill, "description", "")
            examples = _get_card_field(skill, "examples", [])
            examples_str = ""
            if examples:
                examples_str = f"（示例：{'; '.join(str(e) for e in examples)}）"
            if skill_name or skill_desc:
                skill_lines.append(f"{skill_name}：{skill_desc}{examples_str}".strip("："))

        skills_text = "\n  ".join(skill_lines) if skill_lines else "无"
        agents.append(
            f"- `{card_name}` — {card_desc}\n"
            f"  业务能力：\n  {skills_text}"
        )

    return "\n".join(agents) if agents else "（暂无可用业务 Agent）"


def build_system_prompt(
    few_shots: List[Dict],
    agent_cards: Sequence,
) -> str:
    """构建包含业务 Agent 列表与少样本示例的 system prompt。

    Args:
        few_shots: 从 RAG 检索到的相似示例列表
        agent_cards: 可用业务 Agent 的 AgentCard 对象列表

    Returns:
        完整的 system prompt 文本
    """
    schema = IntentResult.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    agents_text = _extract_agents(agent_cards)

    parts = [
        "你是一位多意图识别与任务规划专家，负责将用户 query 解析为可执行的任务规划序列。",
        "",
        _TASK_PLANNING_RULES,
        "",
        "## 可用业务 Agent 列表",
        agents_text,
        "",
        _format_few_shots(few_shots),
        _OUTPUT_RULES.format(schema=schema_str),
    ]
    return "\n".join(parts)
