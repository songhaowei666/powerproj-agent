# 意图识别 Agent 技术规格

## 1. 概述

本文档定义意图识别 Agent 的技术规格，基于 LangChain + LangGraph 构建，负责将用户自然语言 query 解析为多业务意图的任务规划列表。

## 2. 术语

| 术语 | 说明 |
|------|------|
| 统计业务 | 对历史数据进行汇总、分析、报表生成 |
| 规划业务 | 对未来业务进行计划、排期、资源分配 |
| 投资业务 | 对投资组合进行管理、分析、建议 |
| 少样本 | Few-shot prompting，通过相似示例引导 LLM 输出 |

## 3. 输入输出

### 3.1 输入

```json
{
  "query": "帮我统计今年的投资收益，并做下明年的投资规划"
}
```

### 3.2 输出

```json
{
  "tasks": [
    {
      "task_id": "task_1",
      "business": "统计业务",
      "confidence": 0.92,
      "dependencies": [],
      "description": "统计今年所有投资产品的收益情况"
    },
    {
      "task_id": "task_2",
      "business": "规划业务",
      "confidence": 0.88,
      "dependencies": ["task_1"],
      "description": "基于今年收益数据制定明年投资规划"
    }
  ],
  "reasoning": "用户query同时涉及收益统计和未来规划，且规划依赖于统计结果"
}
```

## 4. 模块接口

### 4.1 models.py

```python
class TaskPlan(BaseModel):
    task_id: str
    business: str           # 枚举: 统计业务/规划业务/投资业务
    confidence: float       # [0.0, 1.0]
    dependencies: List[str] # 任务ID列表
    description: str

class IntentResult(BaseModel):
    tasks: List[TaskPlan]
    reasoning: str
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
class IntentAgent:
    def __init__(self, llm: BaseChatModel)
    async def recognize(self, query: str) -> IntentResult
```

> LLM 实例由 `providers/` 目录统一管理，通过构造函数注入 `IntentAgent`，避免各模块重复实例化模型。

## 5. LangGraph 状态图

```
[query] → retrieve_few_shots_node → plan_tasks_node → [IntentResult]
```

### 5.1 State

```python
class IntentState(TypedDict):
    query: str
    few_shots: List[Dict]
    result: Optional[IntentResult]
```

### 5.2 节点说明

| 节点 | 职责 |
|------|------|
| retrieve_few_shots_node | 调用 rag_stub.retrieve_similar_examples，将结果写入 state.few_shots |
| plan_tasks_node | 拼接 system prompt（含 few_shots），调用 LLM with_structured_output(IntentResult)，写入 state.result |

## 6. Prompt 规范

System prompt 模板结构：

```
你是一位多意图识别专家，负责将用户query解析为任务规划列表。

## 业务类型定义
- 统计业务：...
- 规划业务：...
- 投资业务：...

## 少样本示例
{few_shots}

## 输出要求
- 每个任务必须包含 task_id, business, confidence, dependencies, description
- task_id 从 task_1 开始顺序编号
- dependencies 使用 task_id 引用前置任务
- 支持同时识别多个意图

## 输出格式
严格遵循以下 Pydantic 模型：
{schema}
```

## 7. 依赖

- langchain-core
- langchain-openai
- langgraph
- pydantic
- pydantic-settings

## 8. 文件结构

```
config/
├── __init__.py          # 导出 Settings 单例
└── settings.py          # pydantic-settings BaseSettings，读取 .env
providers/
└── llm_provider.py      # 统一模型实例化管理，从 config.settings 读取参数
intent_agent/
├── __init__.py          # 导出 IntentAgent, TaskPlan, IntentResult
├── models.py            # Pydantic 模型定义
├── prompts.py           # System prompt 模板 + 少样本拼接逻辑
├── rag_stub.py          # RAG 查询预留接口（空方法 / mock）
├── graph.py             # LangGraph StateGraph 定义与节点实现
└── agent.py             # IntentAgent 封装类（对外统一入口）
```

## 9. 关键设计决策

1. **LangGraph 状态图仅 2 个节点**：`retrieve → plan`，无路由分支，保持极简
2. **RAG 完全 stub 化**：`rag_stub.py` 返回空列表，后续替换实现即可，不侵入主逻辑
3. **少样本直接拼接 system prompt**：在 `prompts.py` 中组装，不依赖 LangChain 的 FewShotPromptTemplate，保持可控性
4. **配置统一由 `config/` 管理**：`config/settings.py` 基于 `pydantic-settings.BaseSettings` 读取 `.env`，`providers/llm_provider.py` 从中获取 API Key、Base URL、模型名等参数，避免各模块分散读取环境变量
5. **LLM 统一由 `providers/` 管理**：`providers/llm_provider.py` 负责集中实例化 `ChatOpenAI`（或兼容模型），各 Agent 通过构造函数接收 `BaseChatModel` 实例，便于统一切换模型和测试替换

## 10. 后续扩展点

- `rag_stub.retrieve_similar_examples()`：接入 Chroma/Milvus 等向量库
- `models.py`：如需更多任务字段可扩展
- `prompts.py`：业务类型定义可配置化
