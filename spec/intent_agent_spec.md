# 意图识别 Agent 技术规格

## 1. 概述

本文档定义意图识别 Agent 的技术规格，基于 LangChain + LangGraph 构建，负责将用户自然语言 query 解析为可执行的任务规划序列。

改造目标：

- 参考 `examples/agents/task_planning_and_dispatch` 的任务规划与调度设计
- 在提示词中注入各业务 Agent 列表（AgentCard.name / description，skills 仅作业务能力说明）
- 输出从"业务类型任务列表"升级为"带执行顺序的子任务规划"

## 2. 术语

| 术语 | 说明 |
|------|------|
| AgentCard | A2A 协议中描述 Agent 元数据与能力的卡片 |
| Skill | AgentCard 中定义的单个能力，含 `id/name/description/tags/examples` |
| 子任务 | 可由单个业务 Agent 独立完成的最小执行单元 |
| 执行顺序 | 按依赖关系拓扑排序后的子任务 ID 序列 |
| Agent 匹配 | 子任务的 `required_agent` 必须对应某个 AgentCard 的 `name` |

## 3. 输入输出

### 3.1 输入

```json
{
  "query": "查一下北京西500千伏项目的信息，并下载可研设计文件",
  "agent_cards": [
    {
      "name": "planning-agent",
      "skills": [
        {
          "id": "project-query",
          "name": "项目信息查询",
          "description": "根据自然语言查询电力项目基本信息及聚合统计",
          "tags": ["planning", "project", "query"],
          "examples": ["查一下北京西500千伏项目的信息"]
        },
        {
          "id": "file-management",
          "name": "节点文件管理",
          "description": "按节点编码上传、下载、删除文件",
          "tags": ["planning", "file", "upload", "download"],
          "examples": ["下载北京西项目的可研设计文件"]
        }
      ]
    }
  ]
}
```

> `agent_cards` 由调用方（通常是 Main Agent 的 `AgentNetwork`）传入，包含当前可用的全部 A2A Agent 及其 Skill 能力。意图识别 Agent 基于这些真实能力做任务规划，避免产生无法执行的能力匹配。

### 3.2 输出

```json
{
  "task_goal": "查询北京西500千伏项目信息并下载可研设计文件",
  "subtasks": [
    {
      "id": "task_1",
      "name": "查询项目基本信息",
      "description": "根据项目名称查询北京西500千伏项目的基本信息及节点状态",
      "dependencies": [],
      "expected_output": "北京西500千伏项目的基本信息，包括项目编码、电压等级、节点状态等",
      "required_agent": "planning-agent"
    },
    {
      "id": "task_2",
      "name": "下载可研设计文件",
      "description": "根据查询到的项目信息，定位并下载可研设计节点（001）的文件",
      "dependencies": ["task_1"],
      "expected_output": "北京西500千伏项目可研设计节点的文件下载结果或文件链接",
      "required_agent": "planning-agent"
    }
  ],
  "execution_order": ["task_1", "task_2"],
  "reasoning": "用户请求包含项目信息查询和文件下载两个意图；文件下载需要依赖项目信息定位目标项目及节点，因此按 task_1 -> task_2 顺序执行。"
}
```

## 4. 模块接口

### 4.1 models.py

```python
from typing import List
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """子任务定义。"""

    id: str = Field(..., description="子任务编号，如 task_1, task_2")
    name: str = Field(..., description="子任务名称")
    description: str = Field(..., description="子任务详细描述")
    dependencies: List[str] = Field(
        default_factory=list, description="依赖的子任务 ID 列表"
    )
    expected_output: str = Field(..., description="预期输出描述")
    required_agent: str = Field(
        ..., description="目标业务 Agent 名称，必须匹配某个 AgentCard 的 name"
    )


class TaskPlan(BaseModel):
    """任务规划结果。"""

    task_goal: str = Field(..., description="原始任务目标概述")
    subtasks: List[SubTask] = Field(..., description="子任务列表")
    execution_order: List[str] = Field(
        ..., description="按执行顺序排列的子任务 ID"
    )


class IntentResult(BaseModel):
    """意图识别结果。"""

    is_business_query: bool = Field(
        default=True,
        description="是否为电网业务相关查询",
    )
    task_goal: str = Field(..., description="原始任务目标概述")
    subtasks: List[SubTask] = Field(..., description="子任务列表")
    execution_order: List[str] = Field(
        ..., description="按执行顺序排列的子任务 ID"
    )
    reasoning: str = Field(..., description="推理过程说明")
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="需要用户补充信息时的澄清问句",
    )
```

### 4.2 rag_stub.py

```python
async def retrieve_similar_examples(query: str, k: int = 3) -> List[Dict]
```

- 入参：用户 query，返回样本数 k
- 出参：示例列表，每个元素含 `query` 和 `tasks` 字段
- 当前实现：返回空列表 `[]`

### 4.3 agent.py

```python
from typing import Sequence
from langchain_core.language_models import BaseChatModel

class IntentAgent:
    def __init__(self, llm: BaseChatModel)
    async def recognize(self, query: str, agent_cards: Sequence) -> IntentResult
```

- `query`: 用户输入的自然语言查询
- `agent_cards`: 可用业务 Agent 的 AgentCard 对象序列，支持 protobuf 风格与 dataclass 风格
- 返回 `IntentResult`，其中每个子任务的 `required_agent` 必须能匹配到 `agent_cards` 中某个 Agent 的 `name`
- LLM 实例由 `providers/` 目录统一管理，通过构造函数注入 `IntentAgent`

### 4.4 prompts.py

新增能力提取与格式化函数：

```python
def build_system_prompt(
    few_shots: List[Dict],
    agent_cards: Sequence,
) -> str
```

- 从 `agent_cards` 中提取业务 Agent 信息，格式化为 `agent_list` 文本（skills 聚合为业务能力说明）
- 将 `agent_capabilities` 注入 system prompt

## 5. LangGraph 状态图

```
[query, agent_cards] → retrieve_few_shots_node → plan_tasks_node → [IntentResult]
```

### 5.1 State

```python
from typing import TypedDict, List, Dict, Sequence, Optional
from intent_agent.models import IntentResult

class IntentState(TypedDict):
    query: str
    agent_cards: Sequence
    few_shots: List[Dict]
    result: Optional[IntentResult]
```

- `agent_cards` 由外部注入，在 `plan_tasks_node` 中用于构建 system prompt 的能力列表

### 5.2 节点说明

| 节点 | 职责 |
|------|------|
| retrieve_few_shots_node | 调用 `rag_stub.retrieve_similar_examples`，将结果写入 `state.few_shots` |
| plan_tasks_node | 拼接 system prompt（含 `agent_cards` 能力列表与 few_shots），调用 LLM `with_structured_output(IntentResult)`，写入 `state.result` |

## 6. Prompt 规范

System prompt 模板结构：

```
你是一位多意图识别与任务规划专家，负责将用户 query 解析为可执行的任务规划序列。

## 任务分解原则
1. 原子性：每个子任务应足够具体，可由单个 Agent 能力独立完成
2. 完整性：所有子任务的组合必须能够完成原始任务目标
3. 有序性：明确标注子任务之间的依赖关系和执行顺序
4. 可验证性：每个子任务应有明确的完成标准
5. Agent 匹配：每个子任务的 required_agent 必须从下方可用业务 Agent 列表中选取

## 可用 Agent 能力列表
{agent_capabilities}

## 少样本示例
{few_shots}

## 输出要求
- 仔细分析用户 query，识别其中涉及的一个或多个业务意图
- 将每个意图拆分为可由单个 Agent 能力完成的子任务
- 每个子任务必须包含：id, name, description, dependencies, expected_output, required_agent
- id 从 task_1 开始顺序编号，如 task_1, task_2, task_3
- dependencies 使用前置子任务的 id 列表表示依赖关系，无依赖则为空列表 []
- required_agent 必须匹配上方可用业务 Agent 列表中的某个 Agent 名称（AgentCard.name）
- 如果多个子任务之间存在先后依赖关系，请在 dependencies 中正确声明
- description 用一句话概括该子任务的具体内容
- expected_output 说明该子任务完成后应产生的具体结果

## 输出格式
严格遵循以下 JSON Schema：
{schema}

## 用户任务
请根据以上可用 Agent 能力，为以下用户请求进行意图识别与任务分解规划：

{user_query}
```

### 6.1 AgentCard 列表提取格式

从 AgentCard 提取每个业务 Agent，格式如下：

```
- `{agent_name}` — {agent_description}
  业务能力：
  {skill_name}：{skill_description}（示例：...）
```

示例：

```
- `statistics-agent` — 统计业务 Agent，负责电力项目的规模统计、排名分析与指标对比
  业务能力：
  项目统计指标：根据项目编码或项目名称查询具体电力项目...（示例：统计 PRJ001 的指标）
```

## 7. 关键约束

1. **Agent 必须可匹配**：`SubTask.required_agent` 必须是某个 `AgentCard.name`
2. **执行顺序必须有效**：`execution_order` 必须满足所有 `dependencies` 的拓扑约束
3. **支持 protobuf 与 dataclass 两种 AgentCard**：能力提取函数需通过 `getattr` 兼容访问
4. **RAG 完全 stub 化**：`rag_stub.py` 返回空列表，后续替换实现即可，不侵入主逻辑
5. **配置统一由 `config/` 管理**：`config/settings.py` 基于 `pydantic-settings.BaseSettings` 读取 `.env`
6. **LLM 统一由 `providers/` 管理**：`providers/llm_provider.py` 负责集中实例化模型，各 Agent 通过构造函数接收 `BaseChatModel` 实例

## 8. 依赖

- langchain-core
- langchain-openai
- langgraph
- pydantic
- pydantic-settings

## 9. 文件结构

```
config/
├── __init__.py          # 导出 Settings 单例
└── settings.py          # pydantic-settings BaseSettings，读取 .env
providers/
└── llm_provider.py      # 统一模型实例化管理，从 config.settings 读取参数
intent_agent/
├── __init__.py          # 导出 IntentAgent, SubTask, TaskPlan, IntentResult
├── models.py            # Pydantic 模型定义
├── prompts.py           # System prompt 模板 + 能力提取 + 少样本拼接逻辑
├── rag_stub.py          # RAG 查询预留接口（空方法 / mock）
├── graph.py             # LangGraph StateGraph 定义与节点实现
└── agent.py             # IntentAgent 封装类（对外统一入口）
```

## 10. 关键设计决策

1. **LangGraph 状态图保持 2 个节点**：`retrieve → plan`，无路由分支，保持极简
2. **AgentCard 能力注入 prompt**：意图识别阶段即考虑实际可用 Agent 能力，避免规划出无法执行的任务
3. **子任务使用 `required_agent` 路由到业务 Agent**：业务 Agent 对外是黑盒，具体使用哪个内部 Skill 由 Agent 自行决定；主控只负责路由到正确的 endpoint
4. **保留 `reasoning` 字段**：意图识别需要解释多意图拆分与依赖关系的原因
5. **少样本直接拼接 system prompt**：在 `prompts.py` 中组装，不依赖 LangChain 的 FewShotPromptTemplate，保持可控性

## 11. 后续扩展点

- `rag_stub.retrieve_similar_examples()`：接入 Chroma/Milvus 等向量库
- `prompts.py`：能力提取格式可支持更复杂的 AgentCard 字段（如 input_modes、output_modes）
- `models.py`：如需子任务参数、超时、重试等字段可扩展
