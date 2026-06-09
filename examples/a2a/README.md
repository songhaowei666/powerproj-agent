# A2A 示例集合

本目录包含基于 Google A2A 协议的多 Agent 交互示例，涵盖从简单单消息到流式任务推送、再到带 Artifact 返回的完整场景。

---

## single-message — 单消息基础示例

演示最简 A2A Server / Client 交互模式。Server 端接收用户输入后调用 `ChatAgent` 生成回复，并在任务完成后一次性返回完整消息；Client 端通过 `A2ACardResolver` 获取 Agent Card，再发送单条消息并流式接收最终结果。

---

## streaming — 流式推送示例

演示 A2A 流式（SSE）通信的完整链路。Server 端在 `execute` 过程中逐 token 向客户端推送 `TaskStatusUpdateEvent`；Client 端提供两个版本：
- `test_client.py`：直接打印流式文本与 Artifacts。
- `consumer_client.py`：将事件组装为完整 `Task` 对象后交给消费者函数处理。

---

## visit_video_agent — 视频下载 Agent 示例

演示一个带进度反馈和文件 Artifact 返回的实用 Agent。Server 端接收视频 URL，通过 `VideoDownloadAgent` 异步下载，并在下载过程中持续推送进度百分比；下载完成后以 `TaskArtifactUpdateEvent` 形式返回本地文件 URI 与 MIME 类型。Client 端使用 `tqdm` 实时展示下载进度条。

---

## chat_agent.py — 基础聊天 Agent

各示例共用的 LLM 对话封装。基于 `langchain` Prompt Template 与 `providers.llm_provider` 提供的模型，对外暴露 `invoke(query)` 异步生成器，按 token 流式输出文本。

---

## default_server.py — A2A Server 启动辅助

提供 `get_a2a_app()` 与 `create_server()` 快捷函数，封装 `DefaultRequestHandler`、`InMemoryTaskStore`、Agent Card 路由与 JSON-RPC 路由的组装逻辑，供各示例 Server 快速启动 uvicorn 服务。
