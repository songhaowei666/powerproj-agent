# powerproj-agent — Agent 约束文件

本项目是一个基于 **A2A 协议** 的多 Agent 协作系统，采用 **LangChain + LangGraph** 构建核心逻辑，各 Agent 以 FastAPI/Starlette A2A Server 形式独立运行，通过 JSON-RPC over HTTP 进行交互。



## 2. 技术栈

| 层级 | 技术 |
|------|------|
| A2A 协议 | `a2a-sdk` (Google A2A 协议 Python SDK) |
| LLM 框架 | `langchain`, `langgraph` |
| Web 框架 | `starlette` + `uvicorn`（A2A Server） |
| 配置管理 | `pydantic-settings` + `.env` |
| 数据库 | `SQLite`（Planning Agent） |
| HTTP 客户端 | `httpx` |
| 测试 | `pytest`, `pytest-asyncio` |

---

## 3. 通用编码规范

### 3.1 Python 风格

- 类型注解必须完整，尤其对外接口
- 异步优先：`async def` / `await`，避免同步阻塞调用
- 文件顶部写模块级 docstring，说明模块职责

### 3.2 配置管理

所有环境变量、API Key、模型名称统一通过 `config/settings.py` 管理：

```python
from config.settings import settings

# 正确
api_key = settings.openai_api_key

# 错误 —— 禁止各模块自行读取 os.environ
codeapi_key = os.getenv("OPENAI_API_KEY")
```

`config/settings.py` 基于 `pydantic_settings.BaseSettings` 读取项目根目录 `.env` 文件。

### 3.3 LLM 实例化

LLM 统一由 `providers/llm_provider.py` 集中实例化，各 Agent 通过构造函数注入 `BaseChatModel`：

```python
from langchain_core.language_models.chat_models import BaseChatModel

class SomeAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
```

禁止在各 Agent 模块内部直接 `ChatOpenAI(...)` 实例化。

### 3.4 A2A Server 规范

- 所有 Agent Server 统一使用 `a2a_base.py` 提供的 `get_a2a_app()` / `create_server()` 快捷入口，或遵循相同的 Starlette 路由注册方式
- `task.id` 作为 LangGraph 的 `thread_id`，用于状态持久化和中断恢复
- 需要交互确认时，返回 `status.state = "input-required"`
- 执行失败时，返回 `status.state = "failed"`
- 文件下载通过额外路由 `/files/{file_id}` 提供，不在 JSON-RPC 响应中直接返回文件 bytes

### 3.5 错误处理

- 对外 HTTP 调用必须带重试（默认最多 3 次，指数退避）
- 使用 `try/except` 捕获具体异常，禁止裸 `except:`
- 失败信息应包含足够上下文（任务 ID、重试次数、原始异常类型）

---


### 4.1 各模块文件职责（强制）

| 文件 | 职责 |
|------|------|
| `models.py` | **仅**定义 Pydantic / TypedDict 数据模型，禁止包含业务逻辑 |
| `prompts.py` | 存放 prompt 模板和拼接逻辑 |
| `graph.py` | **仅**定义 LangGraph 状态图、节点函数、边路由，禁止直接实例化 LLM / DB |
| `agent.py` | 对外统一入口封装类，负责组装依赖（LLM、DB 等）并暴露简洁接口 |
| `server.py` | A2A HTTP Server，处理 JSON-RPC 请求，调用 graph/agent |
| `database.py` | 数据库访问封装 |
| `__init__.py` | 导出该模块对外暴露的公开接口 |

---

## 5. 关键设计原则

1. **LangGraph 节点保持极简**：每个节点只做一件事，状态通过 `state` 对象传递
2. **中断恢复用 `interrupt` + `Command(resume=...)`**：不引入外部消息队列或 WebSocket
3. **Phase 分层并行**：按 DAG 拓扑分层，同层 `asyncio.gather` 并行，层间串行
4. **任务级熔断**：任一任务重试 3 次仍失败，立即停止后续 Phase


---

## 6. 测试规范

### 6.1 测试文件命名

- `test_{模块名}.py` —— 如 `test_database.py`, `test_graph.py`
- 集成测试：`test_{模块名}_integration.py`

### 6.2 测试原则

- 单元测试：LLM 用 `unittest.mock.MagicMock` / `AsyncMock` 模拟
- 数据库测试：使用临时文件，避免污染生产数据
- 文件测试：使用临时目录，测试后自动清理
- 集成测试：Server 使用 `httpx.AsyncClient` + `asgi-lifespan`

### 6.3 运行测试

```bash
pytest tests/ -v
```

---


> `.env` 文件包含敏感信息，**禁止提交到 git**（已在 `.gitignore` 中配置）。

---

## 8. 规格文档

各 Agent 的详细技术规格见 `spec/` 目录：

- `spec/intent_agent_spec.md` — 意图识别 Agent
- `spec/main_agent_spec.md` — 主控 Agent
- `spec/planning_agent_spec.md` — 规划 Agent

修改 Agent 核心逻辑时，**必须同步更新对应 spec 文档**。
