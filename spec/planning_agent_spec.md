# 规划 Agent (Planning Agent) 技术规格与实现计划

## 1. 概述

规划 Agent 是负责电力项目规划业务的下游业务 Agent，基于 **LangChain + LangGraph + SQLite** 构建，核心能力包括：

1. **项目智能匹配**：根据用户自然语言描述，在 SQLite 项目基本信息表中找到最匹配的电力项目。
2. **多轮交互确认**：通过 A2A `input-required` 状态暂停执行，让用户确认匹配的项目是否正确。
3. **项目信息查询**：支持按自然语言条件查询项目基本信息（名称模糊匹配、线路长度/变电容量范围等），也支持聚合统计查询（如所有项目变电容量的总和、平均值、项目数量等）。
4. **节点文件管理**：针对已确认的项目，支持按节点编码（001 可研设计 / 002 可研评审 / 003 可研批复）进行文件上传、下载、删除。

运行形态：FastAPI A2A Server，暴露 JSON-RPC over HTTP 端点，端口 8001。

---

## 2. 术语

| 术语 | 说明 |
|------|------|
| 项目基本信息表 | SQLite 表 `project_info`，存储项目名称、编码、电压等级、单位编码、线路长度、变电容量 |
| 项目节点文件表 | SQLite 表 `project_node_files`，存储项目编码、节点编码、文件ID 的映射关系 |
| 节点编码 | 001=可研设计，002=可研评审，003=可研批复 |
| 文件存储根目录 | `planning_agent/upload_files/`，按 `项目编码/节点编码/` 子目录组织 |
| 项目确认循环 | 匹配到项目后，通过 `interrupt` 暂停图执行，等待用户确认 |

---

## 3. 数据模型

### 3.1 SQLite 表结构

```sql
-- 项目基本信息表
CREATE TABLE IF NOT EXISTS project_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,        -- 项目名称
    project_code TEXT NOT NULL UNIQUE, -- 项目编码
    voltage_level TEXT,                -- 电压等级：10kv/35kv/220kv/330kv/1000kv
    unit_code TEXT,                    -- 单位编码：01-27（27家省公司）
    line_length REAL,                  -- 线路长度 (km)
    substation_capacity REAL           -- 变电容量 (MVA)
);

-- 项目节点文件表
CREATE TABLE IF NOT EXISTS project_node_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_code TEXT NOT NULL,        -- 项目编码
    node_code TEXT NOT NULL,           -- 节点编码：001/002/003
    file_id TEXT NOT NULL UNIQUE,      -- 文件UUID
    file_name TEXT,                    -- 原始文件名
    file_path TEXT NOT NULL,           -- 本地存储相对路径
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_code) REFERENCES project_info(project_code)
);

-- 全文搜索索引（用于项目名称快速模糊匹配）
CREATE VIRTUAL TABLE IF NOT EXISTS project_info_fts USING fts5(
    project_name, project_code,
    content='project_info',
    content_rowid='id'
);
```

### 3.2 Pydantic 状态模型 (models.py)

```python
class MatchedProject(BaseModel):
    project_name: str
    project_code: str
    voltage_level: Optional[str]    # 电压等级
    unit_code: Optional[str]        # 单位编码
    line_length: Optional[float]
    substation_capacity: Optional[float]
    match_score: float  # 匹配置信度 [0, 1]

class PlanningState(BaseModel):
    query: str = ""                           # 当前完整查询文本
    intent: str = "unknown"                   # 操作类型
    matched_project: Optional[MatchedProject] = None
    project_confirmed: bool = False           # 用户是否已确认项目
    node_code: Optional[str] = None           # 001/002/003
    target_file_id: Optional[str] = None      # 下载/删除时的文件ID
    pending_files: List[Dict[str, Any]] = []  # 从 A2A message 解析出的待上传文件
    result_text: Optional[str] = None         # 执行结果文本
    artifacts: List[Dict[str, Any]] = []      # A2A artifacts
    status: str = "pending"                   # pending / input_required / completed / failed
```

---

## 4. 输入输出

### 4.1 A2A 输入 (tasks/send)

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "id": "task-uuid",
    "sessionId": "session-uuid",
    "message": {
      "role": "user",
      "parts": [
        {"type": "text", "text": "帮我查一下北京西500千伏项目的可研设计文件"},
        {"type": "file", "file": {"bytes": "base64...", "name": "report.pdf"}}
      ]
    }
  },
  "id": 1
}
```

> 上传场景时，`message.parts` 可同时包含 `text`（说明意图和节点）和 `file`（文件内容）。

### 4.2 A2A 输出 (项目待确认)

```json
{
  "id": "task-uuid",
  "status": {
    "state": "input-required",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "找到最匹配的项目：\n名称：北京西500千伏输变电工程\n编码：PRJ001\n线路长度：120km\n变电容量：1000MVA\n\n请问是这个项目吗？请回复'是'或'否'。"}]
    }
  },
  "artifacts": [{"type": "text", "text": "..."}]
}
```

### 4.3 A2A 输出 (查询结果)

```json
{
  "id": "task-uuid",
  "status": {
    "state": "completed",
    "message": {
      "role": "agent",
      "parts": [{"type": "text", "text": "查询结果如下..."}]
    }
  },
  "artifacts": [
    {"type": "text", "text": "【项目信息】\n名称：北京西500千伏输变电工程\n编码：PRJ001\n电压等级：500kv\n单位编码：01\n线路长度：120km\n变电容量：1000MVA"}
  ]
}
```

### 4.4 A2A 输出 (文件下载)

```json
{
  "status": {"state": "completed"},
  "artifacts": [
    {"type": "text", "text": "为您找到以下文件："},
    {"type": "file", "file": {"uri": "http://localhost:8001/files/uuid-xxx", "name": "可研设计报告.pdf"}}
  ]
}
```

---

## 5. 模块接口

### 5.1 database.py

```python
class ProjectDatabase:
    def __init__(self, db_path: str = "planning_agent/planning.db")
    
    # 项目查询
    def search_projects(self, keywords: str, limit: int = 5) -> List[Dict]
    def get_project_by_code(self, code: str) -> Optional[Dict]
    def execute_aggregate_query(self, sql: str) -> Dict  # 执行聚合查询（SUM/AVG/COUNT等）
    
    # 文件记录
    def add_file_record(self, project_code: str, node_code: str, 
                        file_id: str, file_name: str, file_path: str) -> None
    def list_files(self, project_code: str, node_code: Optional[str] = None) -> List[Dict]
    def get_file_by_id(self, file_id: str) -> Optional[Dict]
    def delete_file_record(self, file_id: str) -> bool
    
    # 初始化
    def init_tables(self) -> None
    def seed_data(self) -> None  # 内置种子数据（约10条示例电力项目）
```

### 5.2 project_matcher.py

```python
class ProjectMatcher:
    def __init__(self, db: ProjectDatabase, llm: BaseChatModel)
    
    async def match(self, query: str) -> Optional[MatchedProject]:
        """根据自然语言查询匹配最可能的项目。
        
        实现步骤：
        1. 用 LLM 从 query 中提取关键词和筛选条件（如电压等级、单位编码、线路长度范围、变电容量范围）
        2. 生成并执行 SQLite 查询（LIKE + 等值筛选 + 范围筛选）
        3. 若结果为空，返回 None
        4. 若结果唯一，直接返回
        5. 若结果多条，用 LLM 选择最匹配的一条，并给出 match_score
        """
```

### 5.3 file_manager.py

```python
class FileManager:
    def __init__(self, base_dir: str = "planning_agent/upload_files")
    
    def save_uploaded_file(self, project_code: str, node_code: str, 
                           file_name: str, content_bytes: bytes) -> str:
        """保存上传文件，返回 file_id (UUID)"""
        
    def get_file_path(self, file_id: str) -> Optional[Path]:
        """根据 file_id 获取本地文件路径"""
        
    def delete_file(self, file_id: str) -> bool:
        """删除本地文件"""
        
    def build_download_url(self, file_id: str, base_url: str) -> str:
        """构造文件下载 URL"""
```

### 5.4 graph.py

```python
def build_planning_graph(llm: BaseChatModel, db: ProjectDatabase, fm: FileManager):
    """构建规划 Agent LangGraph。
    
    节点：
    - parse_intent: 解析用户意图和操作类型
    - match_project: 调用 ProjectMatcher 匹配项目
    - confirm_project: interrupt，等待用户确认项目
    - resolve_params: 解析节点编码、文件ID 等参数
    - execute_action: 根据 intent 执行查询/上传/下载/删除
    - finalize: 组装 artifacts 和响应文本
    """
```

### 5.5 executor.py

```python
class PlanningAgentExecutor(AgentExecutor):
    def __init__(self, llm=None, db=None, fm=None)
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None
```

- A2A AgentExecutor 实现，负责接收 A2A 请求、驱动 LangGraph、组装 A2A 响应
- 将 message 中的文件提取为 base64，避免 checkpointer 序列化失败
- 使用 `task.id` 作为 LangGraph `thread_id`
- 检测 graph 中断状态，返回 `input-required` 或 `completed`/`failed`

### 5.6 server.py

- 构造 protobuf AgentCard（`_build_agent_card`）
- 注册静态文件路由 `/files/{file_id}` 供文件下载
- 对外暴露 `AGENT_CARD`、`EXTRA_ROUTES`、`PlanningAgentExecutor`（从 executor.py 导入）

---

## 6. LangGraph 状态图

```
                              ┌─────────────────────────────────────────┐
                              │                                         │
                              ▼                                         │
[START] ──► parse_intent ──► match_project ──► confirm_project ────────┤
                                                  │                     │
                                                  │ (interrupt)         │
                                                  ▼                     │
                                          ┌──────────────┐              │
                                          │ 用户确认项目? │              │
                                          └──────────────┘              │
                                              是 ◄────────┤ 否          │
               │                              │           │             │
               │                              ▼           │             │
               │                    resolve_params         │             │
               │                              │            │             │
               │                              ▼            │             │
               │                    execute_action         │             │
               │                              │            │             │
               │                              ▼            │             │
               │                           finalize ───────┘             │
               │                              │                          │
               │                              ▼                          │
               │                            [END]                         │
               │                                                         │
               └─────────────────────────────────────────────────────────┘
```

### 6.1 节点详细说明

#### parse_intent

使用 LLM 解析用户当前完整 query，识别操作意图：
- `query_project`：查询项目基本信息
- `upload_file`：上传文件到指定节点
- `download_file`：下载/查看指定节点的文件
- `delete_file`：删除指定文件
- `unknown`：意图不明，向用户追问

输出写入 `state.intent`。

#### match_project

调用 `ProjectMatcher.match(state.query)`：
- 若返回 `None`：`state.status = "failed"`，`state.result_text = "未找到匹配的项目..."`
- 若返回项目：写入 `state.matched_project`

#### confirm_project

若 `state.project_confirmed == False` 且已匹配到项目：
```python
interrupt({
    "question": f"找到最匹配的项目：\n名称：{proj.name}\n...\n请问是这个项目吗？"
})
```

恢复后判断用户回复：
- 肯定回答 -> `state.project_confirmed = True`
- 否定回答 -> 清空 `matched_project`，返回 `match_project` 重新匹配或终止

#### resolve_params

根据 `state.intent` 解析附加参数：
- `upload_file` / `download_file` / `delete_file`：从 query 中提取节点编码（001/002/003），写入 `state.node_code`
- `delete_file` / `download_file`：提取文件ID 或文件名，写入 `state.target_file_id`
- `upload_file`：从当前 A2A message 的 parts 中解析 `file` 类型的 part，base64 解码后写入 `state.pending_files`
  - 若用户未附带文件，再次 interrupt 提示用户上传

#### execute_action

```python
if intent == "query_project":
    # 判断是聚合查询还是明细查询
    if is_aggregate_query(state.query):
        # 聚合查询：LLM 生成聚合 SQL 并直接执行
        sql = generate_aggregate_sql(state.query)
        result = db.execute_aggregate_query(sql)
        state.result_text = format_aggregate_result(result)
    else:
        # 明细查询：返回已匹配项目的具体信息
        result = db.get_project_by_code(matched_project.project_code)
        state.result_text = format_project_info(result)
elif intent == "upload_file":
    for f in state.pending_files:
        file_id = fm.save_uploaded_file(...)
        db.add_file_record(...)
    state.result_text = f"成功上传 {len(pending_files)} 个文件到 {node_code} 节点"
elif intent == "download_file":
    files = db.list_files(project_code, node_code)
    state.artifacts = [build_file_artifact(f) for f in files]
elif intent == "delete_file":
    # 删除前二次确认
    if not state.delete_confirmed:
        interrupt({"question": f"确认删除文件 {target_file_name} 吗？该操作不可恢复，请回复'确认删除'或'取消'。"})
        state.delete_confirmed = True
        return state
    db.delete_file_record(target_file_id)
    fm.delete_file(target_file_id)
    state.result_text = "文件已删除"
```

#### finalize

组装最终 `artifacts`：
- 第一个 artifact 为 `result_text` 的 `text` 类型
- 后续附加 `file` 类型 artifacts（下载场景）或 `task_result` 类型

---

## 7. 文件处理说明

### 7.1 上传流程

1. A2A message 中 `parts` 同时包含 `text`（说明意图）和 `file`（base64编码内容）
2. `resolve_params` 节点解析出所有 `file` part，解码 bytes
3. `execute_action` 中按 `upload_files/{project_code}/{node_code}/{filename}` 路径保存
4. 同一项目同一节点上传同名文件时**直接覆盖**（保留最新版本）
5. 向 `project_node_files` 表插入或更新记录（file_id 不变，更新 file_path 和 created_at）

### 7.2 查询与下载流程

1. `execute_action` 查询 `project_node_files` 表
2. 返回文件列表时，**仅返回文件元信息和下载地址**，不返回文件内容 bytes：
   - `text` artifact：文件列表（文件名、节点、上传时间）
   - `file` artifact：下载地址 `http://localhost:8001/files/{file_id}`（URI 方式）
3. FastAPI 注册 `GET /files/{file_id}` 路由，客户端通过该地址下载实际文件内容

### 7.3 删除流程

1. 用户指令如："删除北京西项目可研评审节点的报告.pdf"
2. Agent 先匹配项目并确认，再解析节点编码和文件名
3. 查询 `project_node_files` 获取 `file_id`
4. **二次确认**：通过 `interrupt` 向用户确认"是否删除文件 xxx？"
5. 用户确认后，删除本地文件 + 删除数据库记录

---

## 8. 项目结构

```
planning_agent/
├── __init__.py
├── models.py            # PlanningState, MatchedProject Pydantic 模型
├── database.py          # SQLite 数据库封装
├── project_matcher.py   # 自然语言项目匹配（LLM + SQL）
├── file_manager.py      # 本地文件存取管理
├── graph.py             # LangGraph 状态图定义与节点实现
├── executor.py          # A2A AgentExecutor 实现（LangGraph 驱动 + 响应组装）
├── server.py            # A2A Server（AgentCard + 文件下载路由）
├── client.py            # A2A 客户端（测试用，保留现有）
├── main.py              # 启动入口（保留现有）
└── upload_files/        # 文件存储根目录（.gitignore 忽略）
```

---

## 9. 测试接口

### 9.1 测试范围

| 模块 | 测试重点 | 测试文件 |
|------|----------|----------|
| database.py | 种子数据加载、组合查询、聚合查询、文件记录 CRUD（含覆盖） | `test_database.py` |
| file_manager.py | 文件保存、读取、删除、下载 URL 构建 | `test_file_manager.py` |
| project_matcher.py | 单结果匹配、无结果匹配、多结果匹配 | `test_project_matcher.py` |
| graph.py | 意图解析、聚合查询跳过匹配、graph 编译 | `test_graph.py` |

### 9.2 测试环境

- 数据库使用临时文件，避免污染生产数据
- 文件管理器使用临时目录，测试后自动清理
- LLM 使用 `unittest.mock.MagicMock` + `AsyncMock` 模拟

### 9.3 核心测试用例

```python
# database
- test_seed_data_loaded: 断言种子数据 10 条全部加载
- test_search_by_keywords: 关键词模糊匹配
- test_search_by_voltage_level: 电压等级筛选（如 220kv 返回 3 条）
- test_search_by_unit_code: 单位编码筛选
- test_search_by_line_length_range: 线路长度范围筛选
- test_search_combined_conditions: 名称 + 电压等级组合查询
- test_aggregate_query: SUM/COUNT 聚合查询
- test_file_record_cover: 同名文件覆盖（同一记录更新 file_path）

# file_manager
- test_save_and_get_file: 保存后能通过路径读取
- test_delete_file: 删除后文件不存在
- test_build_download_url: URL 格式正确

# project_matcher
- test_match_single_result: 唯一结果直接返回，match_score=1.0
- test_match_no_result: 无匹配返回 None

# graph
- test_parse_intent_aggregate: 聚合查询 intent=query_project，跳过 match/confirm
- test_graph_compiles: LangGraph 编译成功
```

### 9.4 A2A Server 集成测试（暂不实现代码）

集成测试使用 `httpx.AsyncClient` + `pytest-asyncio` 对 Planning Agent 的 FastAPI Server 进行端到端调用。

#### 测试范围

| 接口 | 方法 | 测试重点 |
|------|------|----------|
| `/.well-known/agent.json` | GET | Agent Card 返回正确，skills 包含 project-query 和 file-management |
| `/` (JSON-RPC tasks/send) | POST | 新任务创建、状态流转、input-required 中断恢复、completed 完成 |
| `/` (JSON-RPC tasks/get) | POST | 根据 task_id 查询已存在的任务 |
| `/` (JSON-RPC tasks/cancel) | POST | 取消任务后状态变为 canceled |
| `/files/{file_id}` | GET | 文件下载路由返回正确的 FileResponse 或 404 |

#### 测试环境

- 使用 `fastapi.testclient.TestClient`（同步）或 `httpx.AsyncClient` + `asgi-lifespan`（异步）
- Server 依赖的 LLM、数据库、文件管理器使用真实实例（集成测试不测 mock）
- 每个测试用例使用独立 task_id，避免任务存储互相污染

#### 核心测试用例

```python
# server 集成测试（test_server_integration.py）

- test_get_agent_card:
    GET /.well-known/agent.json
    断言：status_code == 200
    断言：name == "planning-agent"
    断言：skills 长度 >= 2

- test_tasks_send_new_project_query:
    POST / {jsonrpc: "2.0", method: "tasks/send", params: {message: {parts: [{text: "查一下北京西500千伏项目"}]}}}
    断言：status_code == 200
    断言：result.status.state == "input-required"（首次需要确认项目）
    断言：result.status.message.parts[0].text 包含 "北京西500千伏输变电工程"

- test_tasks_send_confirm_project:
    POST / {jsonrpc: "2.0", method: "tasks/send", params: {id: "同上task_id", message: {parts: [{text: "是的"}]}}}
    断言：result.status.state == "completed"
    断言：result.artifacts[0].parts[0].text 包含 "PRJ001"

- test_tasks_send_aggregate_query:
    POST / {jsonrpc: "2.0", method: "tasks/send", params: {message: {parts: [{text: "所有项目变电容量的总和"}]}}}
    断言：result.status.state == "completed"
    断言：result.artifacts[0].parts[0].text 包含 "变电容量总和"

- test_tasks_send_upload_file:
    POST / {jsonrpc: "2.0", method: "tasks/send", params: {message: {parts: [{text: "上传可研设计文件到北京西项目"}, {file: {bytes: "base64...", name: "design.pdf"}}]}}}
    先确认项目（input-required -> 回复"是"）
    断言最终：result.status.state == "completed"
    断言：result.artifacts[0].parts[0].text 包含 "成功上传"

- test_tasks_send_download_file:
    POST / {jsonrpc: "2.0", method: "tasks/send", params: {message: {parts: [{text: "下载北京西项目的可研设计文件"}]}}}
    先确认项目（input-required -> 回复"是"）
    断言最终：result.status.state == "completed"
    断言：artifacts 中存在 file 类型 part，uri 包含 "/files/"

- test_tasks_get:
    POST / {jsonrpc: "2.0", method: "tasks/get", params: {id: "已存在的task_id"}}
    断言：result.id == task_id
    断言：result.status.state 为 completed / input-required / failed 之一

- test_tasks_cancel:
    POST / {jsonrpc: "2.0", method: "tasks/cancel", params: {id: "已存在的task_id"}}
    断言：result.status.state == "canceled"

- test_download_file:
    GET /files/{file_id}
    断言：status_code == 200
    断言：headers["content-disposition"] 包含文件名

- test_download_file_not_found:
    GET /files/not-exist-uuid
    断言：status_code == 404
```

## 10. 已确认设计决策

1. **种子数据**：`database.py` 内置约 10 条示例电力项目种子数据，启动时自动初始化。
2. **查询复杂度**：支持数值组合查询，包括电压等级、单位编码、线路长度范围、变电容量范围的多条件组合。
3. **删除确认**：删除文件前通过 `interrupt` 二次确认，用户明确回复后才执行删除。
4. **文件覆盖**：同一项目同一节点上传同名文件时直接覆盖，保留最新版本。
