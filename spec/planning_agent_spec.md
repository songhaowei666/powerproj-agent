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
      "parts": [
        {
          "text": "找到最匹配的项目：\n名称：北京西500千伏输变电工程\n编码：PRJ001\n...\n\n请问是这个项目吗？"
        },
        {
          "mediaType": "application/vnd.powerproj.confirmation+json",
          "data": {
            "type": "confirmation",
            "action": "project_confirm",
            "title": "请确认项目",
            "options": [
              {"id": "yes", "label": "是", "replyText": "是"},
              {"id": "no", "label": "否", "replyText": "否"}
            ]
          }
        }
      ]
    }
  }
}
```

> Web 客户端根据 `data.options` 渲染按钮；用户点击后发送 `replyText` 作为 text part 恢复任务。删除确认使用 `action=delete_confirm`，选项为「确认删除 / 取消」。

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
    
    # 初始化（内部方法，由构造器自动调用）
    def _init_tables(self) -> None
    def _seed_data(self) -> None  # 内置种子数据（约10条示例电力项目）
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

### 9.1 测试目录结构

```
planning_agent/tests/
├── unit/                    # 单元测试：使用 Mock LLM，覆盖各模块核心逻辑
│   ├── test_database.py
│   ├── test_file_manager.py
│   ├── test_project_matcher.py
│   ├── test_graph.py
│   ├── test_executor.py
│   └── test_server.py       # Server 端到端测试，使用 Mock LLM
└── functional/              # 功能测试：使用真实 LLM，默认跳过（需 RUN_FUNCTIONAL_TESTS=1）
    └── test_server.py
```

### 9.2 测试范围

| 模块 | 测试重点 | 测试文件 |
|------|----------|----------|
| database.py | 种子数据加载、组合查询、聚合查询、文件记录 CRUD（含覆盖）、边界查询 | `unit/test_database.py` |
| file_manager.py | 文件保存、读取、删除、空目录清理、下载 URL 构建 | `unit/test_file_manager.py` |
| project_matcher.py | 单结果匹配、无结果匹配、多结果 LLM 选择、非法编码 fallback | `unit/test_project_matcher.py` |
| graph.py | 意图解析、聚合/明细查询、项目确认、上传/下载/删除、graph 编译、辅助函数 | `unit/test_graph.py` |
| executor.py | 文件提取、取消任务、空消息失败、聚合完成、中断恢复、失败状态 | `unit/test_executor.py` |
| server.py | Agent Card、SendMessage、GetTask、CancelTask、文件下载路由 | `unit/test_server.py` + `functional/test_server.py` |

### 9.3 测试环境

- 数据库使用临时文件，避免污染生产数据
- 文件管理器使用临时目录，测试后自动清理
- 单元测试 LLM 使用 `unittest.mock.MagicMock` + `AsyncMock` 模拟
- 功能测试使用 `providers.llm_provider.get_llm()` 真实实例，通过 `RUN_FUNCTIONAL_TESTS=1` 环境变量启用

### 9.4 核心测试用例

```python
# database (unit/test_database.py)
- test_seed_data_loaded: 断言种子数据 10 条全部加载
- test_search_by_keywords: 关键词模糊匹配
- test_search_by_voltage_level: 电压等级筛选
- test_search_by_unit_code: 单位编码筛选
- test_search_by_line_length_range: 线路长度范围筛选
- test_search_by_substation_capacity_range: 变电容量范围筛选
- test_search_combined_conditions: 名称 + 电压等级组合查询
- test_aggregate_query: SUM/COUNT 聚合查询
- test_aggregate_query_empty_result: 聚合查询无结果时的返回
- test_file_record_cover: 同名文件覆盖（同一记录更新 file_path）
- test_list_files_without_node_code: 不带 node_code 查询项目下所有文件
- test_get_file_by_id / test_get_file_by_name: 文件记录查询
- test_delete_file_record_returns_false_when_not_found: 删除不存在记录返回 False

# file_manager (unit/test_file_manager.py)
- test_save_and_get_file: 保存后能通过路径读取
- test_save_overwrites_existing_file: 同名文件覆盖
- test_delete_file_by_location: 按位置删除文件
- test_delete_file_and_cleanup_empty_dirs: 删除后清理空目录
- test_get_file_path_finds_existing_file / returns_none_when_empty: get_file_path 行为
- test_build_download_url: URL 格式正确

# project_matcher (unit/test_project_matcher.py)
- test_match_single_result: 唯一结果直接返回，match_score=1.0
- test_match_no_result: 无匹配返回 None
- test_match_multiple_results_uses_llm_selector: 多结果时调用 LLM 选择
- test_match_multiple_results_fallback_to_first_when_invalid_code: LLM 返回非法编码时回退到第一条

# graph (unit/test_graph.py)
- test_graph_compiles: LangGraph 编译成功
- test_parse_intent_aggregate: 聚合查询跳过 match/confirm
- test_parse_intent_aggregate_fallback_keywords: 聚合判断 LLM 异常时回退到关键词匹配
- test_parse_intent_exception_fallback: parse_intent LLM 异常时 intent 回退为 unknown
- test_parse_intent_detail_query: 明细查询的项目确认与恢复
- test_confirm_project_negative: 用户否定项目后任务失败
- test_confirm_project_vague_response: 用户模糊回答时 project_confirmed 保持 False
- test_confirm_project_already_confirmed: 项目已确认时 confirm_project 直接返回
- test_match_project_no_result: 未匹配到项目时任务失败
- test_detail_query_project_not_found_in_db: 明细查询数据库无记录时失败
- test_detail_query_no_matched_project: 明细查询未匹配项目时失败
- test_upload_file_flow / test_download_file_flow / test_delete_file_flow: 文件操作完整流程
- test_upload_file_no_matched_project / no_node_code / no_pending_files: 上传文件各类失败分支
- test_download_file_no_matched_project / no_files: 下载文件失败/无文件分支
- test_delete_file_no_matched_project / by_id / no_target / not_found: 删除文件各类分支
- test_delete_file_cancelled: 删除操作用户取消
- test_unknown_intent: 未知意图导致失败
- test_unsupported_intent: 不支持的操作类型导致失败
- test_finalize_no_result_text: finalize 默认结果文本
- test_aggregate_query_drop_sql_raises / insert_sql_raises: 非法聚合 SQL 抛出 ValueError

# executor (unit/test_executor.py)
- test_extract_single_file / empty_message / skip_text_part / ignores_exception: 文件提取
- test_cancel_sends_cancelled_message: 取消任务
- test_execute_empty_message_fails: 空消息失败
- test_execute_aggregate_query_completes: 聚合查询完成
- test_execute_returns_input_required_on_interrupt: 中断恢复状态
- test_execute_graph_interrupt_exception: ainvoke 抛出 GraphInterrupt 仍返回 input-required
- test_execute_interrupt_info_missing_fallback: interrupt 信息缺失时使用默认提示
- test_execute_failed_state: 失败状态处理

# server (unit/test_server.py，使用 Mock LLM)
- test_get_agent_card: Agent Card 路由
- test_tasks_send_new_project_query / confirm_project / aggregate_query / upload_file / download_file: SendMessage 状态流转
- test_tasks_get / test_tasks_cancel: GetTask / CancelTask
- test_download_file / test_download_file_not_found: 文件下载路由（测试用自定义路由）
- TestServerDownloadFile: 直接测试 planning_agent.server.download_file handler（DB 无记录 / 磁盘缺失 / 正常返回）

# server (functional/test_server.py，使用真实 LLM，默认跳过)
- test_get_agent_card
- test_aggregate_query
- test_project_query_and_confirm
- test_upload_and_download_file
```

### 9.5 A2A Server 功能测试（functional/test_server.py）

功能测试站在用户角度，使用真实 LLM 对 Planning Agent Server 进行端到端 JSON-RPC 调用，验证完整业务链路。

#### 测试边界

**覆盖范围：**
- A2A Server 的 JSON-RPC 端到端调用（Agent Card、SendMessage、GetTask、CancelTask）
- 真实 LLM 对用户自然语言的意图解析与项目匹配
- 真实 SQLite 数据库和本地文件系统的读写
- 核心业务场景：项目查询、聚合统计、文件上传/下载

**不覆盖范围：**
- 使用 Mock LLM 的分支场景（由单元测试覆盖）
- 单元级别的异常处理分支（如 LLM 异常 fallback、文件不存在等）
- 外部依赖的网络故障、并发压力、安全注入等非功能场景

#### 测试环境

- 使用 `starlette.testclient.TestClient`（同步）
- LLM 使用 `providers.llm_provider.get_llm()` 真实实例
- 数据库和文件管理器使用临时文件/目录，每个测试独立隔离
- 默认通过 `pytestmark = pytest.mark.skipif(...)` 跳过，避免 CI 产生意外费用；设置 `RUN_FUNCTIONAL_TESTS=1` 后启用

#### 核心测试用例

```python
# functional/test_server.py

class TestAgentCard:
    def test_get_agent_card(self, test_client):
        """GET /.well-known/agent-card.json 返回正确的 Agent Card。"""
        断言：status_code == 200
        断言：name == "planning-agent"
        断言：skills 长度 >= 2

class TestAggregateQuery:
    def test_aggregate_query(self, test_client):
        """聚合查询直接返回 completed，结果包含变电容量相关信息。"""
        SendMessage: "所有项目变电容量的总和是多少"
        断言：status.state == "TASK_STATE_COMPLETED"
        断言：artifacts[0].parts[0].text 包含 "变电" / "容量" / "总和"

class TestProjectQueryAndConfirm:
    def test_project_query_and_confirm(self, test_client):
        """查询项目后用户确认，返回 completed 项目详情。"""
        第一轮 SendMessage: "查一下北京西500千伏输变电工程"
        断言：status.state == "TASK_STATE_INPUT_REQUIRED"
        第二轮 SendMessage（taskId=同上）: "是的"
        断言：status.state == "TASK_STATE_COMPLETED"
        断言：artifacts[0].parts[0].text 包含 "PRJ001" 或 "北京西"

class TestFileUploadAndDownload:
    def test_upload_and_download_file(self, test_client):
        """上传文件后经下载路由获取原文件内容。"""
        第一轮 SendMessage（含 base64 文件）: "上传可研设计文件到北京西500千伏输变电工程"
        断言：status.state == "TASK_STATE_INPUT_REQUIRED"
        第二轮 SendMessage（taskId=同上）: "是的"
        断言：status.state == "TASK_STATE_COMPLETED"
        第三轮 SendMessage: "下载北京西500千伏输变电工程可研设计文件"
        断言：status.state == "TASK_STATE_INPUT_REQUIRED"
        第四轮 SendMessage（taskId=同上）: "是的"
        断言：status.state == "TASK_STATE_COMPLETED"
        从 artifacts 中提取 file_id
        GET /files/{file_id}
        断言：status_code == 200
        断言：resp.content == 原始文件 bytes
```

## 10. 已确认设计决策

1. **种子数据**：`database.py` 内置约 10 条示例电力项目种子数据，启动时自动初始化。
2. **查询复杂度**：支持数值组合查询，包括电压等级、单位编码、线路长度范围、变电容量范围的多条件组合。
3. **删除确认**：删除文件前通过 `interrupt` 二次确认，用户明确回复后才执行删除。
4. **文件覆盖**：同一项目同一节点上传同名文件时直接覆盖，保留最新版本。
