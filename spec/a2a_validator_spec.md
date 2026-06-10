# A2A 服务验证器 (A2A Validator) 技术规格

## 1. 概述

A2A 服务验证器是一个独立的诊断与验证工具，基于 `a2a-sdk` 客户端与 **Streamlit** 构建，用于对任意符合 Google A2A 协议的服务端点进行自动化健康检查与功能验证。

核心能力：
1. **连通性探测**：验证 `/.well-known/agent.json` 端点可达
2. **Agent Card 解析**：使用 `A2ACardResolver` 获取并校验 Agent Card 结构
3. **单消息测试**：以非流式模式发送测试消息，验证请求/响应链路
4. **流式测试**：以 SSE 流式模式发送测试消息，验证实时推送与 Artifact 组装
5. **可视化报告**：通过 Streamlit 界面展示逐项验证结果与详情

运行形态：独立 Streamlit 应用，通过命令 `streamlit run a2a_validator/app.py` 启动。

---

## 2. 术语

| 术语 | 说明 |
|------|------|
| 验证项 (Check Item) | 单次验证流程中的独立检查点：`connectivity`、`agent_card`、`single_message`、`streaming` |
| 验证状态 | `passed`（通过） / `failed`（失败） / `skipped`（跳过） / `pending`（待执行） |
| Agent Card | A2A 协议定义的 Agent 元数据，通过 `GET /.well-known/agent.json` 获取 |
| 单消息模式 | `ClientConfig(streaming=False)`，服务端一次性返回完整结果 |
| 流式模式 | `ClientConfig(streaming=True)`，服务端通过 SSE 逐块推送 `TaskStatusUpdateEvent` |

---

## 3. 输入输出

### 3.1 验证器输入

由 Streamlit 界面收集：

```json
{
  "base_url": "http://localhost:9999",
  "test_message": "你好，请简单回复"
}
```

### 3.2 验证器输出

`A2AValidator.validate()` 返回结构：

```json
{
  "base_url": "http://localhost:9999",
  "connectivity": {
    "status": "passed",
    "detail": "服务可连通"
  },
  "agent_card": {
    "status": "passed",
    "detail": "成功解析 Agent Card: 流式Chat服务智能体",
    "data": "<AgentCard protobuf 对象>"
  },
  "single_message": {
    "status": "passed",
    "detail": "收到 3 条响应",
    "data": ["<SendMessageResponse>", "..."]
  },
  "streaming": {
    "status": "passed",
    "detail": "收到 42 个文本片段, 1 个 artifact",
    "data": {
      "chunks": ["你好", "！", "很高兴", "..."],
      "artifacts": ["<Artifact protobuf 对象>"]
    }
  }
}
```

> 任一前置项 `failed` 时，后续依赖项仍保持 `pending` 状态，不会执行。

---

## 4. 模块接口

### 4.1 validator.py

```python
class A2AValidator:
    def __init__(self, base_url: str, timeout: float = 30.0)
    """
    初始化验证器。
    
    Args:
        base_url: A2A 服务根地址，如 http://localhost:9999
        timeout: HTTP 请求超时时间（秒），默认 30.0
    """

    async def validate(self, test_message: str = "你好，请简单回复") -> Dict[str, Any]
    """
    执行完整验证流程，依次检查 connectivity → agent_card → single_message → streaming。
    
    Args:
        test_message: 用于发送测试的消息内容
    
    Returns:
        包含各验证项状态、详情与原始数据的字典
    """
```

内部私有方法（不对外暴露）：

```python
async def _check_connectivity(self, client: httpx.AsyncClient) -> None
async def _check_agent_card(self, client: httpx.AsyncClient) -> Any  # 返回 AgentCard 或 None
async def _check_single_message(self, client: httpx.AsyncClient, agent_card: Any, message_text: str) -> None
async def _check_streaming(self, client: httpx.AsyncClient, agent_card: Any, message_text: str) -> None
```

### 4.2 app.py

Streamlit 单页应用，无对外 Python API，核心交互逻辑：

1. 通过 `st.text_input` 收集 `base_url` 与 `test_message`
2. 点击 `st.button("开始验证")` 后，实例化 `A2AValidator`
3. 使用 `asyncio.run(validator.validate(...))` 执行验证
4. 根据 `results` 中各验证项的 `status`，分别渲染 `st.success` / `st.error` / `st.warning`
5. 通过 `st.expander` 提供 Agent Card、流式片段、单消息响应的详情查看

---

## 5. 验证流程

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  输入 URL   │────►│  输入测试消息   │────►│  点击开始验证   │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                     ▼
                                         ┌─────────────────────┐
                                         │  _check_connectivity │
                                         │  GET /.well-known/...│
                                         └──────────┬──────────┘
                                                    │ failed?
                                          是 ◄──────┤──────► 否
               │                                    │
               │                                    ▼
               │                         ┌─────────────────────┐
               │                         │  _check_agent_card   │
               │                         │  A2ACardResolver     │
               │                         └──────────┬──────────┘
               │                                    │ failed?
               │                          是 ◄──────┤──────► 否
               │               │                    │
               │               │                    ▼
               │               │         ┌─────────────────────┐
               │               │         │ _check_single_message│
               │               │         │ streaming=False      │
               │               │         └──────────┬──────────┘
               │               │                    │ failed?
               │               │            是 ◄────┤─────► 否
               │               │    │               │
               │               │    │               ▼
               │               │    │    ┌─────────────────────┐
               │               │    │    │  _check_streaming   │
               │               │    │    │  streaming=True      │
               │               │    │    │  检查 capabilities   │
               │               │    │    └──────────┬──────────┘
               │               │    │               │
               ▼               ▼    ▼               ▼
         ┌─────────────────────────────────────────────────────┐
         │              Streamlit 渲染验证结果面板              │
         └─────────────────────────────────────────────────────┘
```

### 5.1 各检查点详细说明

#### connectivity

- 请求方法：`GET {base_url}/.well-known/agent.json`
- 通过条件：HTTP 状态码为 `200`
- 失败条件：网络不可达、DNS 解析失败、HTTP 错误状态码

#### agent_card

- 使用 `A2ACardResolver(httpx_client=client, base_url=base_url)` 解析
- 通过条件：成功返回 `AgentCard` protobuf 对象
- 失败条件：端点返回非 JSON、解析异常、protobuf 反序列化失败

#### single_message

- 配置 `ClientConfig(streaming=False)`
- 构造 `Message(role=ROLE_USER, parts=[Part(text=test_message)])`
- 通过条件：`send_message` 迭代器成功返回至少一条响应，无异常抛出
- 失败条件：请求超时、服务端返回 JSON-RPC error、网络中断

#### streaming

- 前置检查：`agent_card.capabilities.streaming` 必须为 `True`，否则标记为 `skipped`
- 配置 `ClientConfig(streaming=True)`
- 通过条件：成功接收到 `TaskStatusUpdateEvent`（含文本片段）或 `TaskArtifactUpdateEvent`
- 失败条件：SSE 连接断开、服务端未推送任何事件、请求异常

---

## 6. 依赖

- `a2a-sdk`（Google A2A 协议 Python SDK）
- `httpx`
- `streamlit`
- `uuid`（标准库）

---

## 7. 文件结构

```
a2a_validator/
├── __init__.py          # 导出 A2AValidator
├── validator.py         # 核心验证逻辑（A2A 客户端封装 + 四项检查）
└── app.py               # Streamlit 界面入口
```

---

## 8. 测试接口

### 8.1 测试范围

| 模块 | 测试重点 | 测试文件 |
|------|----------|----------|
| validator.py | 四项检查独立通过/失败/跳过逻辑、异常处理、结果组装 | `test_validator.py` |

### 8.2 测试环境

- `httpx.AsyncClient` 使用 `respx` 或 `unittest.mock.AsyncMock` 进行 HTTP 层 mock
- `a2a-sdk` 的 `A2ACardResolver`、`ClientFactory` 使用 `unittest.mock.patch` 进行 mock
- 无需真实 A2A Server，全部在单元测试中通过 mock 覆盖

### 8.3 核心测试用例

测试文件位于 `a2a_validator/tests/test_validator.py`。

| 测试函数 | 场景 | 断言要点 |
|----------|------|----------|
| `test_validate_all_passed` | 四项检查全部通过 | `connectivity/agent_card/single_message/streaming` 均为 `passed`；`agent_card.data` 为返回的 `AgentCard` 对象 |
| `test_validate_connectivity_failed` | 连通性检查失败（网络不可达） | `connectivity.status == "failed"`；`agent_card/single_message/streaming` 保持 `pending`（短路） |
| `test_validate_agent_card_failed` | Agent Card 解析失败（返回非法 JSON） | `agent_card.status == "failed"`；`single_message/streaming` 保持 `pending` |
| `test_validate_single_message_failed` | 单消息请求超时 | `single_message.status == "failed"`；异常详情包含 `ReadTimeout` |
| `test_validate_streaming_skipped` | 服务端声明不支持流式 | `streaming.status == "skipped"`；详情包含"不支持流式传输" |
| `test_validate_streaming_failed` | 流式请求返回 HTTP 500 | `streaming.status == "failed"`；异常详情包含 `HTTPStatusError` |
| `test_validate_well_known_non_200` | `/.well-known/agent.json` 返回 404 | `connectivity.status == "failed"`；详情包含状态码 `404` |

### 8.4 Streamlit 界面测试（可选）

使用 `streamlit.testing.v1.AppTest` 对 `app.py` 进行自动化 UI 测试（需 streamlit >= 1.35）。

| 测试函数 | 场景 | 断言要点 |
|----------|------|----------|
| `test_app_render` | 验证页面控件正确渲染 | 存在 2 个 `text_input`（默认值分别为 `http://localhost:9999` 和 `你好，请简单回复`）和 1 个 `button`（标签为"开始验证"） |

---

## 9. 关键设计决策

1. **纯客户端验证器**：`a2a_validator` 不启动任何 Server，仅作为 A2A 协议的客户端对目标服务进行探测。
2. **前置失败短路**：`connectivity` 或 `agent_card` 任一失败，后续 `single_message` / `streaming` 不再执行，避免无意义请求。
3. **流式能力按需跳过**：通过读取 `AgentCard.capabilities.streaming` 动态决定是否执行流式测试，而非对所有服务强制测试 SSE。
4. **异常信息包含类型名**：所有 `except Exception` 捕获的异常详情中均包含 `type(exc).__name__`，便于快速定位问题类别。
5. **Streamlit 状态驱动渲染**：界面无复杂状态管理，完全由 `validator.validate()` 返回的结果字典驱动，通过 `st.success` / `st.error` / `st.warning` 直接映射四种状态。
