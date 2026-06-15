"""System prompt 模板 + AgentCard 能力提取 + 少样本拼接逻辑。"""

import json
from typing import List, Dict, Sequence

from intent_agent.models import IntentResult


_TASK_PLANNING_RULES = """\
## 任务分解原则
1. 原子性：每个子任务应足够具体，可由单个 Agent 能力独立完成
2. 完整性：所有子任务的组合必须能够完成原始任务目标
3. 有序性：明确标注子任务之间的依赖关系和执行顺序
4. 可验证性：每个子任务应有明确的完成标准
5. 能力匹配：每个子任务的 required_capability 必须从上方可用能力列表中选取
"""

_OUTPUT_RULES = """\
## 输出要求
- 仔细分析用户 query，识别其中涉及的一个或多个业务意图
- 将每个意图拆分为可由单个 Agent 能力完成的子任务
- 每个子任务必须包含：id, name, description, dependencies, expected_output, required_capability
- id 从 task_1 开始顺序编号，如 task_1, task_2, task_3
- dependencies 使用前置子任务的 id 列表表示依赖关系，无依赖则为空列表 []
- required_capability 必须匹配上方可用能力列表中的某个 skill id
- 如果多个子任务之间存在先后依赖关系，请在 dependencies 中正确声明
- description 用一句话概括该子任务的具体内容
- expected_output 说明该子任务完成后应产生的具体结果
- 对每个子任务给出 confidence（0.0 ~ 1.0），表示对该子任务识别的置信度

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


def _extract_capabilities(agent_cards: Sequence) -> str:
    """从 AgentCard 列表中提取并格式化能力描述。

    Args:
        agent_cards: A2A AgentCard 对象列表或序列化 dict 列表

    Returns:
        格式化后的能力描述文本
    """
    capabilities: List[str] = []
    for card in agent_cards:
        card_name = _get_card_field(card, "name", "未知 Agent")
        skills = _get_card_field(card, "skills", [])
        for skill in skills:
            skill_id = _get_card_field(skill, "id", "")
            skill_name = _get_card_field(skill, "name", "")
            skill_desc = _get_card_field(skill, "description", "")
            tags = _get_card_field(skill, "tags", [])
            tags_str = ", ".join(str(t) for t in tags) if tags else "无"
            examples = _get_card_field(skill, "examples", [])
            examples_str = ""
            if examples:
                examples_str = f"\n  示例：{'; '.join(str(e) for e in examples)}"

            capabilities.append(
                f"- `{skill_id}` — {skill_name}\n"
                f"  所属Agent：{card_name}\n"
                f"  描述：{skill_desc}\n"
                f"  标签：{tags_str}{examples_str}"
            )

    return "\n".join(capabilities) if capabilities else "（暂无可用 Agent 能力）"


def build_system_prompt(
    few_shots: List[Dict],
    agent_cards: Sequence,
) -> str:
    """构建包含 AgentCard 能力列表与少样本示例的 system prompt。

    Args:
        few_shots: 从 RAG 检索到的相似示例列表
        agent_cards: 可用业务 Agent 的 AgentCard 对象列表

    Returns:
        完整的 system prompt 文本
    """
    schema = IntentResult.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    capabilities_text = _extract_capabilities(agent_cards)

    parts = [
        "你是一位多意图识别与任务规划专家，负责将用户 query 解析为可执行的任务规划序列。",
        "",
        _TASK_PLANNING_RULES,
        "",
        "## 可用 Agent 能力列表",
        capabilities_text,
        "",
        _format_few_shots(few_shots),
        _OUTPUT_RULES.format(schema=schema_str),
    ]
    return "\n".join(parts)
