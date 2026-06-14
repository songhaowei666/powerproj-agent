# 主控 Agent (Main Agent) 技术规格

## 1. 概述

主控 Agent 是用户请求的**统一入口**，基于 LangChain + LangGraph 构建，负责：

1. **Agent 网络管理**：通过 `AgentNetwork` 发现和维护下游业务 Agent，统一拉取各 Agent 的 AgentCard
2. **意图识别**：调度意图识别 Agent，基于当前 Agent 网络中的全部 AgentCard 对 user query 进行意图解析与任务规划
3. **置信度检查与补全循环**：当任务为空或任意子任务置信度 < 0.8 时，向用户发起澄清提问，收到补充信息后重新识别，直到满足条件
4. **分阶段并行调度**：根据子任务依赖关系构建 DAG，按拓扑分层，同层任务并行执行，层间串行
5. **失败重试与熔断**：每个任务最多重试 3 次，任一任务最终失败即终止整个流程并返回错误
6. 透传业务 Agent 返回的文本与文件下载链接

运行形态：与其他业务 Agent 一致，为 FastAPI A2A Server，暴露 JSON-RPC over HTTP 端点。

## 2. 术语

| 术语 | 说明 |
|------|------|
| 业务 Agent | 下游具体执行业务的 Agent：statistics-agent、planning-agent、investment-agent |
| AgentNetwork | 主控 Agent 维护的 Agent 注册表与发现组件，负责拉取并缓存 AgentCard |
| Skill | AgentCard 中定义的单个能力，含 `id/name/description/tags/examples` |
| SubTask | 意图识别输出的子任务，包含 `id/name/description/dependencies/expected_output/required_capability` |
| Phase | 执行阶段。按依赖拓扑分层后，同一层无依赖关系的任务构成一个 Phase |
| 置信度补全循环 | 当识别结果不满足置信度阈值时，通过 `interrupt` 暂停图执行，等待用户补充信息后恢复并重新识别 |
| 任务熔断 | 某个任务重试 3 次仍失败后，立即停止后续所有 Phase 的执行 |

## 3. 输入输出

### 3.1 A2A 输入 (tasks/send)

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "id": "task-uuid",
    "sessionId": "session-uuid",
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "查一下北京西500千伏项目的信息，并下载可研设计文件"}]
    }
  },
  "id": 1
}
```

### 3.2 A2A 输出 (正常完成)

```json
{
  "id": "task-uuid",
  "sessionId": "session-uuid",
  "status": {
    "state": "completed",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "根据您的需求，我已完成项目信息查询和文件下载..."}]
    }
  },
  "artifacts": [
    {
      "type": "text",
      "text": "【总结】\n\n根据您的需求，我已完成项目信息查询和文件下载。\n\n1. 项目信息：北京西500千伏项目，项目编码 XXX，电压等级 500kV...\n2. 文件下载：可研设计节点文件已获取，链接如下..."
    },
    {
      "type": "task_result",
      "task_id": "task_1",
      "required_capability": "project-query",
      "artifacts": [
        {"type": "text", "text": "项目信息原文..."}
      ]
    },
    {
      "type": "task_result",
      "task_id": "task_2",
      "required_capability": "file-management",
      "artifacts": [
        {"type": "file", "url": "http://xxx/design.pdf", "name": "可研设计.pdf"}
      ]
    }
  ]
}
```

> **说明**：第一个 artifact 是 LLM 生成的**自然语言总结**，后续 `task_result` 类型的 artifacts 保留各业务 Agent 的原始返回，供用户追溯完整详情。

### 3.3 A2A 输出 (需要用户补充信息)

```json
{
  "id": "task-uuid",
  "status": {
    "state": "input-required",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "以下任务置信度较低，请补充相关信息：..."}]
    }
  },
  "artifacts": [
    {"type": "text", "text": "以下任务置信度较低，请补充相关信息：..."}
  ]
}
```

> 客户端收到 `input-required` 后，应再次调用 `tasks/send`（相同 `id`），在 `message.parts` 中携带用户的补充信息。

### 3.4 A2A 输出 (执行失败)

```json
{
  "id": "task-uuid",
  "status": {
    "state": "failed",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "任务 task_2 执行失败（已重试3次）：连接超时"}]
    }
  },
  "artifacts": [
    {"type": "text", "text": "任务 task_2 执行失败（已重试3次）：连接超时"}
  ]
}
```

## 4. 模块接口

### 4.1 models.py

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from intent_agent.models import IntentResult, SubTask


class TaskOutput(BaseModel):
    """单个子任务的执行结果。"""

    task_id: str
    required_capability: str
    status: str           # "success" | "failed"
    artifacts: List[Dict[str, Any]] = []
    error: Optional[str] = None


class MainState(BaseModel):
    """Main Agent LangGraph 状态。"""

    query: str = ""
    session_id: Optional[str] = None
    intent_result: Optional[IntentResult] = None
    phases: List[List[str]] = []       # 拓扑分层结果，每个元素是同层 subtask id 列表
    current_phase_idx: int = 0
    task_outputs: Dict[str, TaskOutput] = {}
    failed_task_id: Optional[str] = None
    error_message: Optional[str] = None
    final_artifacts: List[Dict[str, Any]] = []
    status: str = "pending"             # pending | executing | completed | failed
    summary: Optional[str] = None
```

### 4.2 agent_network.py

```python
from typing import List, Sequence
import httpx
from a2a.types import AgentCard


class AgentNetwork:
    """A2A Agent 网络管理器。

    负责注册 Agent endpoint、拉取并缓存 AgentCard、向意图识别模块提供统一能力视图。
    """

    def __init__(self, timeout: float = 10.0)
    def register(self, url: str) -> None
    def register_from_config(self, urls: Sequence[str]) -> None
    async def discover(self) -> List[AgentCard]
    def get_cards(self) -> List[AgentCard]
    async def aclose(self) -> None
```

- `register`：注册一个 A2A Agent endpoint URL
- `register_from_config`：从配置批量注册默认 URL
- `discover`：并发拉取所有已注册 Agent 的 `/.well-known/agent.json`，缓存并返回 AgentCard 列表
- `get_cards`：返回最近一次 `discover` 的缓存结果
- 单个 AgentCard 拉取失败时记录日志，不影响其他 Agent

### 4.3 registry.py

```python
from typing import List

DEFAULT_AGENT_URLS: List[str] = [
    "http://localhost:8001",  # planning-agent
    "http://localhost:8002",  # investment-agent
    "http://localhost:8003",  # statistics-agent
]
```

- 从 `business -> url` 的硬编码映射改为**默认 Agent URL 列表**
- 主控 Agent 启动时使用 `AgentNetwork.register_from_config(DEFAULT_AGENT_URLS)` 注册
- 后续可扩展为动态服务发现（如 Consul、环境变量）

### 4.4 executor.py

```python
async def call_business_agent(
    subtask: SubTask,
    agent_cards: List[AgentCard],
    session_id: str,
) -> Dict[str, Any]:
    """
    调用下游业务 Agent 的 A2A JSON-RPC 接口。

    内部行为：
    - 根据 subtask.required_capability 在 agent_cards 中查找匹配的 Skill
    - 通过 Skill 所属 AgentCard 的 supported_interfaces 获取 endpoint URL
    - 构造 tasks/send JSON-RPC 请求
    - 最多重试 3 次（含指数退避）
    - 返回 {"status": "success", "artifacts": [...]}

    异常：
    - 找不到匹配的 Skill 或 endpoint 时抛出 ValueError
    - 超过 3 次仍失败时抛出 Exception，由 graph 节点捕获并触发熔断
    """
```

### 4.5 agent_executor.py

```python
class MainAgentExecutor(AgentExecutor):
    def __init__(self, llm: BaseChatModel, agent_network: AgentNetwork)
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None
```

- 实现 A2A SDK 的 `AgentExecutor` 接口，由 `DefaultRequestHandler` 调用
- 负责从 `context.message` 提取文本、调用 LangGraph
- 会话历史直接从 A2A SDK 管理的 `task.history` 中提取，无需 Executor 自行维护
- 使用 `TaskUpdater` 向客户端发送 `input-required`、`completed`、`failed` 等状态
- 将 `MainState.final_artifacts` 转换为 protobuf `Part` 列表后发送

### 4.6 graph.py

```python
def build_main_graph(
    llm: BaseChatModel,
    agent_network: AgentNetwork,
) -> CompiledStateGraph:
    """
    构建主控 LangGraph。

    节点：
    - recognize_and_check: 从 AgentNetwork 获取 AgentCard，调用 intent_agent，检查置信度，不足时 interrupt 等待用户补充
    - build_phases: 根据 dependencies 拓扑排序分 Phase
    - execute_current_phase: 并行调用当前 Phase 的所有业务 Agent
    - finalize: 组装最终结果或失败信息
    - summarize: 生成自然语言总结

    边：
    - START -> recognize_and_check -> build_phases -> execute_current_phase
    - execute_current_phase --(conditional)--> execute_current_phase (下一层)
    - execute_current_phase --(conditional)--> finalize (完成或失败)
    - finalize -> summarize -> END
    """
```

### 4.7 server.py

```python
if __name__ == "__main__":
    create_server(
        agent_executor=_agent_executor,
        agent_card=AGENT_CARD,
        port=8000,
        lifespan=lifespan,
        log_level="info",
    )
```

- 使用 `a2a_base.create_server` 启动 A2A 服务
- `create_server` 内部使用 SDK 的 `DefaultRequestHandler` 处理 JSON-RPC 协议
- 启动时创建全局 `AgentNetwork` 实例
- 通过 `lifespan` 在启动时调用 `agent_network.discover()` 预加载所有 AgentCard
- 将 `agent_network` 注入 `MainAgentExecutor` 与 `build_main_graph(llm, agent_network)`

## 5. LangGraph 状态图

```
                              ┌─────────────────────────────────────┐
                              │                                     │
                              ▼                                     │
[START] ──► recognize_and_check ──► build_phases ──► execute_current_phase
                                                           │
                                                           │ (conditional)
                                                           ▼
                                                    ┌──────────────┐
                                                    │  还有 Phase?  │
                                                    └──────────────┘
                                              是 ◄──────┤ 否
               │                              │         │
               │                              ▼         ▼
               │                    execute_current_phase   finalize
               │                              │               │
               │                              │               ▼
               │                              │            summarize
               │                              │               │
               │                              │               ▼
               │                              │              [END]
               │                              │
               └──────────────────────────────┘
```

### 5.1 节点详细说明

#### recognize_and_check

```python
async def recognize_and_check(state: MainState) -> MainState:
    agent_cards = await agent_network.discover()

    while True:
        result = await intent_agent.recognize(state.query, agent_cards)
        state.intent_result = result

        if not result.subtasks:
            question = "您的请求不够明确，请补充更多细节，例如您想查询、统计、规划还是投资？"
            clarification = interrupt({"question": question})
            state.query += f"\n补充信息：{clarification}"
            continue

        low_conf_tasks = [t for t in result.subtasks if t.confidence < 0.8]
        if low_conf_tasks:
            descs = "、".join([f"{t.description}（置信度{t.confidence:.2f}）" for t in low_conf_tasks])
            question = f"以下任务置信度较低，请补充相关信息：{descs}"
            clarification = interrupt({"question": question})
            state.query += f"\n补充信息：{clarification}"
            continue

        break
    return state
```

> **关键机制**：`interrupt` 暂停图执行后，外部收到 `input-required`；用户再次发送补充信息，外部使用 `Command(resume=...)` 恢复；节点会**重新完整执行**，此时 `state.query` 已追加补充信息，`intent_agent.recognize()` 会重新识别，形成循环。
>
> **AgentCard 来源**：每次进入该节点时从 `AgentNetwork` 获取最新 AgentCard，确保能感知 Agent 能力变化。

#### build_phases

1. 从 `state.intent_result.subtasks` 构建 DAG
2. 使用 Kahn 算法拓扑排序并**分层**
3. 每层（入度同时降为 0 的一批任务）构成一个 Phase
4. 写入 `state.phases` 和 `state.current_phase_idx = 0`
5. 若检测到环，退化为单 Phase（所有任务放一层）

#### execute_current_phase

1. 取出 `state.phases[state.current_phase_idx]` 中的 subtask id 列表
2. 通过 id 查找到对应的 `SubTask`
3. 使用 `asyncio.gather(*coros, return_exceptions=True)` 并行调用所有业务 Agent
4. 遍历结果：
   - 成功：写入 `state.task_outputs[tid]`
   - 失败（任意一个）：设置 `state.failed_task_id`、`state.error_message`、`state.status = "failed"`，立即返回
5. 全部成功：`state.current_phase_idx += 1`

#### finalize

- 若 `state.status == "failed"`：`final_artifacts` 为错误文本
- 若完成：按 Phase 顺序汇总所有 `task_outputs` 的 `artifacts` 到 `final_artifacts`

#### summarize

- 调用 LLM 对所有成功的 `task_outputs` 生成一段自然语言总结
- 输入：各子任务的 `required_capability`、`description`、以及 `artifacts` 中的 `text` 内容
- 输出：`state.summary`（字符串）
- 文件链接单独提取，以引用列表形式附在总结末尾，不交给 LLM 处理内容
- `final_artifacts` 第一个元素为总结文本，后续为原始 `task_result` 结构

### 5.2 条件路由

```python
def route_after_execution(state: MainState) -> str:
    if state.status == "failed":
        return "finalize"
    if state.current_phase_idx >= len(state.phases):
        return "finalize"
    return "execute_current_phase"
```

## 6. 并行执行策略

### 6.1 Phase 内并行

同一 Phase 内的所有任务**无依赖关系**，使用 `asyncio.gather` 并行发起 A2A HTTP 请求。

```python
coros = [
    call_business_agent(subtask_map[tid], agent_cards, session_id)
    for tid in phase_task_ids
]
results = await asyncio.gather(*coros, return_exceptions=True)
```

### 6.2 Phase 间串行

不同 Phase 之间存在依赖（如 Phase 1 的输出是 Phase 2 的输入），必须等待前一 Phase 全部完成后才能进入下一 Phase。

## 7. AgentNetwork 发现策略

### 7.1 启动时预发现

```python
network = AgentNetwork()
network.register_from_config(DEFAULT_AGENT_URLS)
await network.discover()
graph = build_main_graph(llm, network)
```

### 7.2 每次请求时刷新

在 `recognize_and_check` 节点中调用 `await agent_network.discover()`，可感知 Agent 上下线变化。代价是每次请求增加 N 次 HTTP 调用。

### 7.3 推荐折中

- 启动时 `discover()` 一次
- 提供 `refresh()` 接口供外部触发刷新
- 每次请求使用 `get_cards()` 读取缓存
- 执行阶段若找不到对应 Skill，再尝试 `discover()` 刷新

## 8. 重试与容错

### 8.1 单任务重试

`executor.call_business_agent` 内部实现：

```python
for attempt in range(3):
    try:
        resp = await httpx.AsyncClient().post(url, json=payload)
        # 检查 HTTP 状态码和 JSON-RPC error
        return {"status": "success", "artifacts": ...}
    except Exception:
        if attempt == 2:
            raise
        await asyncio.sleep(1 * (attempt + 1))  # 线性退避：1s, 2s
```

### 8.2 AgentCard 发现容错

- `AgentNetwork.discover()` 中单个 Agent 拉取失败仅记录日志，不抛异常
- 至少一个 AgentCard 发现成功即可继续
- 若全部失败，意图识别节点可中断并提示"当前无可用 Agent"

### 8.3 熔断策略

- `execute_current_phase` 中，`asyncio.gather(return_exceptions=True)` 收集所有结果
- 若任一任务返回 Exception，立即设置失败状态并返回
- **不再执行后续 Phase**
- `finalize` 节点将失败信息写入 `final_artifacts`
- 最终 A2A 响应 `status.state = "failed"`

## 9. 依赖

- langchain-core
- langchain-openai
- langgraph (>= 0.2, 需支持 `interrupt` 与 `Command`)
- langgraph-checkpoint
- pydantic
- fastapi
- httpx
- uvicorn

## 10. 文件结构

```
main_agent/
├── __init__.py          # 导出关键类
├── agent_executor.py    # MainAgentExecutor：实现 A2A AgentExecutor 接口
├── agent_network.py     # AgentNetwork：AgentCard 发现与缓存
├── models.py            # MainState, TaskOutput Pydantic 模型
├── registry.py          # 默认 Agent URL 列表
├── executor.py          # 下游业务 Agent A2A 客户端封装 + 重试逻辑
├── graph.py             # LangGraph 状态图定义与节点实现
└── server.py            # 启动入口，使用 a2a_base.create_server
```

> `server.py` 不再自行实现 JSON-RPC 路由，而是通过 `a2a_base.create_server` 复用 SDK 的 `DefaultRequestHandler`，降低协议实现成本。

## 11. 关键设计决策

1. **AgentNetwork 集中管理 AgentCard**：主控 Agent 统一发现下游 Agent 能力，避免意图识别 Agent 硬编码业务类型与 URL，实现能力驱动的任务规划。
2. **LangGraph `interrupt` 实现同步阻塞式交互**：利用 LangGraph 内置的 `interrupt` / `Command(resume=...)` 机制，在 `recognize_and_check` 节点中暂停图执行，无需外部消息队列或 WebSocket。
3. **单节点内循环完成意图重识别**：`recognize_and_check` 使用 `while True` 封装"识别 → 检查 → 中断 → 恢复 → 重新识别"的完整循环，避免在图中增加多余的回环边。
4. **Phase 分层执行实现依赖调度**：不引入复杂的工作流引擎，仅用 Kahn 算法对 DAG 分层，`asyncio.gather` 实现层内并行，条件边实现层间串行。
5. **任务级熔断**：任一任务 3 次重试失败后立即终止整个流程，不继续执行后续 Phase，确保错误及时暴露。
6. **文件下载链接直接透传**：业务 Agent 返回的 `file` 类型 artifact 原样放入 `final_artifacts`，主控 Agent 不做下载、合并、存储。
7. **总结节点后置**：所有业务 Agent 执行完毕后，由 LLM 统一生成自然语言总结，文件链接以引用列表附后。总结与原始结果同时返回，兼顾可读性与信息完整性。
8. **MemorySaver 作为 Checkpoint**：使用内存型 checkpoint 保存图状态，以 `task.id` 作为 `thread_id`，支持中断恢复。当前为单实例内存存储，后续可替换为持久化 checkpoint（如 Postgres、Redis）。
