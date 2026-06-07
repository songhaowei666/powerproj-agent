"""System prompt 模板 + 少样本拼接逻辑。"""

import json
from typing import List, Dict

from intent_agent.models import IntentResult


_BUSINESS_DEFINITIONS = """\
## 业务类型定义

- **统计业务**：对历史数据进行汇总、分析、报表生成。例如：统计收益、计算平均值、生成月度报表。
- **规划业务**：对未来业务进行计划、排期、资源分配。例如：制定年度计划、安排项目里程碑、分配预算。
- **投资业务**：对投资组合进行管理、分析、建议。例如：股票买卖建议、基金配置、风险评估。
"""

_OUTPUT_RULES = """\
## 输出要求

- 仔细分析用户 query，识别其中涉及的一个或多个业务意图
- 每个任务必须包含：task_id, business, confidence, dependencies, description
- task_id 从 task_1 开始顺序编号，如 task_1, task_2, task_3
- dependencies 使用前置任务的 task_id 列表表示依赖关系，无依赖则为空列表 []
- confidence 为 0.0 ~ 1.0 的浮点数，表示对该意图识别的置信度
- 支持同时识别多个意图，每个意图对应一个独立的任务
- 如果多个任务之间存在先后依赖关系（如规划依赖于统计数据），请在 dependencies 中正确声明
- description 用一句话概括该任务的具体内容

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


def build_system_prompt(few_shots: List[Dict]) -> str:
    """构建包含少样本示例的 system prompt。

    Args:
        few_shots: 从 RAG 检索到的相似示例列表

    Returns:
        完整的 system prompt 文本
    """
    schema = IntentResult.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    parts = [
        "你是一位多意图识别专家，负责将用户 query 解析为任务规划列表。",
        "",
        _BUSINESS_DEFINITIONS,
        "",
        _format_few_shots(few_shots),
        _OUTPUT_RULES.format(schema=schema_str),
    ]
    return "\n".join(parts)
