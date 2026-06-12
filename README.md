# powerproj-agent

基于 [Google A2A 协议](https://github.com/google/a2a) 的多 Agent 协作系统，面向电网投资、规划、统计等业务场景。核心编排采用 **LangChain + LangGraph**，各 Agent 以独立 A2A Server 运行，通过 JSON-RPC over HTTP 通信。

## 系统架构

用户请求统一进入 **主控 Agent (Main Agent)**，由它调用意图识别模块解析任务，再按依赖关系分阶段调度下游业务 Agent，最后汇总各任务结果返回。

```mermaid
flowchart TB
    User["用户 / A2A Client"]
    Main["Main Agent :8000"]
    Intent["Intent Agent（内嵌）"]
    Stats["Statistics Agent :8003"]
    Plan["Planning Agent :8001"]
    Invest["Investment Agent :8002"]

    User -->|tasks/send| Main
    Main --> Intent
    Main -->|Phase 并行调度| Stats
    Main -->|Phase 并行调度| Plan
    Main -->|Phase 并行调度| Invest
    Stats --> Main
    Plan --> Main
    Invest --> Main
    Main -->|completed / input-required / failed| User
```

### 核心流程

1. **意图识别**：将自然语言 query 解析为带依赖关系的任务列表（统计 / 规划 / 投资）
2. **置信度补全**：任务为空或任意任务置信度 < 0.8 时，通过 `interrupt` 暂停并返回 `input-required`，等待用户补充后重新识别
3. **分阶段并行**：按 DAG 拓扑分层，同层任务 `asyncio.gather` 并行，层间串行
4. **失败熔断**：单任务最多重试 3 次，任一任务最终失败则终止后续 Phase
5. **结果汇总**：LLM 生成自然语言总结，并保留各业务 Agent 原始 artifacts

## Agent 一览

| Agent | 目录 | 端口 | 说明 |
|-------|------|------|------|
| 主控 Agent | `main_agent/` | 8000 | 用户统一入口，任务编排与结果聚合 |
| 规划 Agent | `planning_agent/` | 8001 | 电力项目匹配、信息查询、节点文件管理（SQLite） |
| 投资 Agent | `investment_agent/` | 8002 | 投资分析（当前为固定测试返回） |
| 统计 Agent | `statistics_agent/` | 8003 | 数据统计分析（当前为固定测试返回） |
| 意图识别 | `intent_agent/` | — | 内嵌于主控 Agent，非独立服务 |
| A2A 验证器 | `a2a_validator/` | — | Streamlit 诊断工具，验证任意 A2A 端点 |

业务类型与 endpoint 映射见 `main_agent/registry.py`：

| 业务类型 | Endpoint |
|----------|----------|
| 统计业务 | `http://localhost:8003` |
| 规划业务 | `http://localhost:8001` |
| 投资业务 | `http://localhost:8002` |

## 技术栈

| 层级 | 技术 |
|------|------|
| A2A 协议 | `a2a-sdk` |
| LLM 框架 | `langchain`, `langgraph` |
| Web 框架 | `starlette` / `fastapi` + `uvicorn` |
| 配置管理 | `pydantic-settings` + `.env` |
| 数据库 | SQLite（Planning Agent） |
| HTTP 客户端 | `httpx` |
| 验证工具 UI | `streamlit` |
| 测试 | `pytest`, `pytest-asyncio` |

## 目录结构

```
powerproj-agent/
├── a2a_base.py           # A2A Server 快捷入口（get_a2a_app / create_server）
├── a2a_validator/        # A2A 协议验证工具（Streamlit）
├── config/               # 全局配置（pydantic-settings）
├── intent_agent/         # 意图识别 Agent（LangGraph）
├── main_agent/           # 主控 Agent（编排调度）
├── planning_agent/       # 规划 Agent（SQLite + 文件管理）
├── investment_agent/     # 投资 Agent
├── statistics_agent/     # 统计 Agent
├── providers/            # LLM 统一实例化
├── rag/                  # RAG 基础设施（预留）
├── examples/             # A2A 调用示例与演示脚本
├── spec/                 # 各 Agent 技术规格文档
├── tests/                # 集成测试
└── AGENTS.md             # Agent 开发约束与编码规范
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd powerproj-agent

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖（项目暂未提供 requirements.txt，以下为常用包）
pip install a2a-sdk langchain langchain-openai langgraph pydantic-settings \
    fastapi uvicorn starlette httpx pytest pytest-asyncio streamlit \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://your-api-base/v1   # 可选，OpenAI 兼容接口地址
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
```

所有配置通过 `config/settings.py` 统一读取，各模块禁止直接访问 `os.environ`。

### 3. 启动服务

各 Agent 需分别启动。建议先启动业务 Agent，再启动主控 Agent：

```bash
# 终端 1 - 规划 Agent
python planning_agent/main.py

# 终端 2 - 投资 Agent
python investment_agent/main.py

# 终端 3 - 统计 Agent
python statistics_agent/main.py

# 终端 4 - 主控 Agent（用户入口）
python main_agent/server.py
```

启动后可访问各 Agent 的 Agent Card：

```bash
curl http://localhost:8000/.well-known/agent.json   # 主控
curl http://localhost:8001/.well-known/agent.json   # 规划
curl http://localhost:8002/.well-known/agent.json   # 投资
curl http://localhost:8003/.well-known/agent.json   # 统计
```

### 4. 发送请求

通过 A2A JSON-RPC `tasks/send` 向主控 Agent 发送请求：

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/send",
    "params": {
      "id": "task-001",
      "sessionId": "session-001",
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "帮我统计今年的投资收益，并做下明年的投资规划"}]
      }
    },
    "id": 1
  }'
```

响应状态说明：

| state | 含义 | 客户端行为 |
|-------|------|-----------|
| `completed` | 执行完成 | 读取 `artifacts` 获取结果 |
| `input-required` | 需要补充信息 | 用相同 `id` 再次 `tasks/send`，携带补充内容 |
| `failed` | 执行失败 | 读取错误信息 |

### 5. A2A 协议验证

使用 Streamlit 验证器对任意 A2A 端点进行健康检查：

```bash
streamlit run a2a_validator/app.py
```

验证项包括：连通性探测、Agent Card 解析、单消息测试、流式测试。

## 规划 Agent 能力

规划 Agent 是功能最完整的业务 Agent，核心能力包括：

- **项目智能匹配**：根据自然语言在 SQLite 项目库中匹配电力项目
- **多轮交互确认**：匹配结果需用户确认（`input-required` + `interrupt` 恢复）
- **项目信息查询**：支持明细查询与聚合统计（如变电容量总和、项目数量等）
- **节点文件管理**：按节点编码（001 可研设计 / 002 可研评审 / 003 可研批复）上传、下载、删除文件
- **文件下载路由**：`GET /files/{file_id}`

详细规格见 [`spec/planning_agent_spec.md`](spec/planning_agent_spec.md)。

## 示例代码

`examples/` 目录提供 A2A 协议交互参考：

| 目录 | 说明 |
|------|------|
| `examples/a2a/single-message/` | 单消息基础 Server / Client |
| `examples/a2a/streaming/` | SSE 流式推送完整链路 |
| `examples/a2a/visit_video_agent/` | 带进度反馈和文件 Artifact 的 Agent |
| `examples/agents/task_planning_and_dispatch/` | 任务规划与分发参考 |

详见 [`examples/a2a/README.md`](examples/a2a/README.md)。

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行 Planning Agent 单元测试
pytest planning_agent/tests/ -v

# 运行 A2A 验证器测试
pytest a2a_validator/tests/ -v
```

## 规格文档

各 Agent 的详细技术规格见 `spec/` 目录：

| 文档 | 说明 |
|------|------|
| [`spec/main_agent_spec.md`](spec/main_agent_spec.md) | 主控 Agent：编排流程、LangGraph 状态图、A2A 接口 |
| [`spec/intent_agent_spec.md`](spec/intent_agent_spec.md) | 意图识别：任务规划模型、Prompt 规范 |
| [`spec/planning_agent_spec.md`](spec/planning_agent_spec.md) | 规划 Agent：数据模型、节点流程、文件管理 |
| [`spec/a2a_validator_spec.md`](spec/a2a_validator_spec.md) | A2A 验证器：验证项与输出结构 |

修改 Agent 核心逻辑时，需同步更新对应 spec 文档与测试。开发约束详见 [`AGENTS.md`](AGENTS.md)。

## 设计原则

- **LangGraph 节点极简**：每个节点只做一件事，状态通过 `state` 对象传递
- **中断恢复**：使用 `interrupt` + `Command(resume=...)` 实现多轮交互，不引入外部消息队列
- **Phase 分层并行**：DAG 拓扑分层，同层并行、层间串行
- **任务级熔断**：单任务重试 3 次仍失败，立即停止后续 Phase
- **LLM 集中管理**：统一由 `providers/llm_provider.py` 实例化，禁止各模块自行创建模型客户端
- **配置集中管理**：统一通过 `config/settings.py` 读取 `.env`，禁止模块内直接读环境变量

## 注意事项

- `.env` 含敏感信息，已加入 `.gitignore`，请勿提交到版本库
- 统计 Agent 与投资 Agent 当前返回固定测试数据，用于验证编排链路
- 意图识别 Agent 的 RAG 少样本检索（`intent_agent/rag_stub.py`）当前返回空列表，后续可接入 `rag/` 模块
