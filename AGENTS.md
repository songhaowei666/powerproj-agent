# powerproj-agent — Agent 约束文件

本项目是一个基于 **A2A 协议** 的多 Agent 协作系统，采用 **LangChain + LangGraph** 构建核心逻辑，各 Agent 以 FastAPI/Starlette A2A Server 形式独立运行，通过 JSON-RPC over HTTP 进行交互。

## 1. 命名
- 函数动词开头；布尔量用 is_/has_ 前缀
- 内部方法以 _ 开头，例如 _xx 

## 2. 目录

| 目录 | 用途 |
|------|------|
| `a2a_validator/` | A2A 协议验证工具，提供协议合规性校验 |
| `config/` | 全局配置管理，基于 `pydantic-settings` 统一读取 `.env` 环境变量 |
| `examples/` | 示例代码，包含各 Agent 的  调用示例和演示脚本,用于正式开发的ai coding参考 |
| `intent_agent/` | **意图识别 Agent**，负责解析用户输入意图并输出结构化任务列表及其依赖关系 |
| `investment_agent/` | **投资分析 Agent**，负责电网投资业务的实现 |
| `main_agent/` | **主控 Agent**，编排调度各子 Agent，负责任务分发与结果聚合 |
| `planning_agent/` | **规划 Agent**，负责电网规划业务的实现 |
| `providers/` | LLM 提供商统一封装，集中实例化各模型客户端 |
| `rag/` | RAG（检索增强生成）基础设施，提供向量检索与知识库支持 |
| `memory/` | Agent Memory，跨轮次会话记忆与上下文推断（见 [spec/agent_memory_spec.md](./spec/agent_memory_spec.md)） |
| `spec/` | 各 Agent 的详细技术规格文档（Markdown） |
| `statistics_agent/` | **统计 Agent**，负责电网统计业务的实现 |


## 3. 技术栈

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

## 4. 通用编码规范

- 类型注解必须完整，尤其对外接口
- 异步优先：`async def` / `await`，避免同步阻塞调用
- 文件顶部写模块级 docstring，说明模块职责
- 所有环境变量、API Key、模型名称统一通过 `config/settings.py` 管理：
- LLM 统一由 `providers/` 集中实例化，各 Agent 通过构造函数注入
- 对外 HTTP 调用必须带重试（默认最多 3 次，指数退避）
- 使用 `try/except` 捕获具体异常，禁止裸 `except:`
- 失败信息应包含足够上下文（任务 ID、重试次数、原始异常类型）

---

## 7. 测试规范

- 命名：`test_{模块名}.py` —— 如 `test_database.py`, `test_graph.py`
- 单元测试：所有数据都模拟，不要访问真实数据；确保每一行代码都被单元测试涉及；内部方法不单独测试，只通过外部方法测试；如果某个内部方法无法通过外部方法测试所有分支，请及时告诉我；数据库测试使用临时文件，避免污染数据；文件测试：使用临时目录，测试后自动清理
- 单元测试放入 unit/  ，功能测试放入 functional/ 
- 功能测试：不要使用mock数据，站在用户的角度进行，直接访问真实的目录，一切贴合真实的场景；如果测试失败，先不修改代码，首先检查单元测试，并回复用户失败原因；

---

## 用法 

- 修改 Agent 核心逻辑时，**必须同步更新对应 spec 文档 和 对应的测试**。
- 每个py脚本尽量不要超过1000行，遵循良好的设计风格。
