"""A2A 服务验证器核心逻辑。

参考 examples/a2a 中的客户端实现，封装对 A2A 服务的自动化验证流程。
"""

from typing import Dict, Any
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Role,
    Part,
    SendMessageRequest,
    SendMessageConfiguration,
)
from a2a.helpers import get_message_text


class A2AValidator:
    """验证指定 A2A 服务的可用性与功能完整性。

    Args:
        base_url: A2A 服务根地址，如 ``http://localhost:9999``
        timeout: HTTP 请求超时时间（秒）
    """

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.results: Dict[str, Any] = {}
        self.last_task_id: str | None = None

    async def validate(self, test_message: str = "你好，请简单回复") -> Dict[str, Any]:
        """执行完整验证流程。

        依次检查：
        1. 基础连通性（Agent Card 端点可访问）
        2. Agent Card 解析
        3. 单消息（non-streaming）发送与响应
        4. 流式（streaming）发送与响应

        Args:
            test_message: 用于发送测试的消息内容

        Returns:
            包含各检查项结果的字典
        """
        self.results = {
            "base_url": self.base_url,
            "connectivity": {"status": "pending", "detail": ""},
            "agent_card": {"status": "pending", "detail": "", "data": None},
            "single_message": {"status": "pending", "detail": "", "data": None},
            "streaming": {"status": "pending", "detail": "", "data": None},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # 1. 基础连通性
            await self._check_connectivity(client)
            if self.results["connectivity"]["status"] == "failed":
                return self.results

            # 2. Agent Card
            agent_card = await self._check_agent_card(client)
            if agent_card is None:
                return self.results

            # 3. 单消息测试
            await self._check_single_message(client, agent_card, test_message)

            # 4. 流式消息测试
            await self._check_streaming(client, agent_card, test_message)

        return self.results

    async def send_message(
        self,
        message_text: str,
        task_id: str | None = None,
    ) -> Dict[str, Any]:
        """向 A2A 服务发送单条消息（支持指定 task_id 进行多轮对话）。

        Args:
            message_text: 消息内容
            task_id: 指定任务 ID（用于继续已有对话）；为 None 时自动创建新任务

        Returns:
            包含响应结果的字典
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._send_raw_message(client, message_text, task_id)

    async def _send_raw_message(
        self,
        client: httpx.AsyncClient,
        message_text: str,
        task_id: str | None = None,
    ) -> Dict[str, Any]:
        """底层：直接发送 JSON-RPC SendMessage 请求。"""
        import google.protobuf.json_format as json_format
        from a2a.types.a2a_pb2 import SendMessageRequest, SendMessageConfiguration

        req = SendMessageRequest()
        if task_id:
            # 继续已有对话：显式指定 task_id
            req.message.task_id = task_id
        req.message.message_id = uuid4().hex
        req.message.role = Role.ROLE_USER
        req.message.parts.add().text = message_text
        req.configuration.CopyFrom(SendMessageConfiguration())

        payload = {
            "jsonrpc": "2.0",
            "method": "SendMessage",
            "params": json_format.MessageToDict(req),
            "id": 1,
        }
        resp = await client.post(
            self.base_url + "/",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "A2A-Version": "1.0",
            },
        )
        resp.raise_for_status()
        rpc_body = resp.json()

        if rpc_body.get("error"):
            return {
                "status": "failed",
                "detail": f"JSON-RPC Error {rpc_body['error'].get('code')}: {rpc_body['error'].get('message')}",
                "data": rpc_body,
            }

        rpc_result = rpc_body.get("result", {})
        task_state = rpc_result.get("status", {}).get("state", "")
        state_name = (
            task_state.replace("TASK_STATE_", "").lower()
            if isinstance(task_state, str)
            else str(task_state)
        )

        # 记录服务端返回的 task_id
        returned_task_id = rpc_result.get("id", task_id or "")
        self.last_task_id = returned_task_id

        return {
            "status": "passed",
            "detail": f"Task 状态: {state_name}",
            "data": rpc_result,
            "task_id": returned_task_id,
        }

    async def _check_connectivity(self, client: httpx.AsyncClient) -> None:
        try:
            # a2a-sdk 默认注册路径为 /.well-known/agent-card.json
            resp = await client.get(f"{self.base_url}/.well-known/agent-card.json")
            if resp.status_code == 200:
                self.results["connectivity"] = {
                    "status": "passed",
                    "detail": "服务可连通",
                }
            else:
                self.results["connectivity"] = {
                    "status": "failed",
                    "detail": f"Agent Card 端点返回状态码 {resp.status_code}",
                }
        except Exception as exc:  # noqa: BLE001
            self.results["connectivity"] = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
            }

    async def _check_agent_card(self, client: httpx.AsyncClient) -> Any:
        try:
            resolver = A2ACardResolver(
                httpx_client=client, base_url=self.base_url
            )
            card = await resolver.get_agent_card()
            self.results["agent_card"] = {
                "status": "passed",
                "detail": f"成功解析 Agent Card: {card.name}",
                "data": card,
            }
            return card
        except Exception as exc:  # noqa: BLE001
            self.results["agent_card"] = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "data": None,
            }
            return None

    async def _check_single_message(
        self,
        client: httpx.AsyncClient,
        agent_card: Any,
        message_text: str,
    ) -> None:
        result = await self._send_raw_message(client, message_text)
        self.results["single_message"] = result

    async def _check_streaming(
        self,
        client: httpx.AsyncClient,
        agent_card: Any,
        message_text: str,
    ) -> None:
        try:
            capabilities = getattr(agent_card, "capabilities", None)
            if not capabilities or not capabilities.streaming:
                self.results["streaming"] = {
                    "status": "skipped",
                    "detail": "Agent Card 声明不支持流式传输",
                    "data": None,
                }
                return

            config = ClientConfig(httpx_client=client, streaming=True)
            factory = ClientFactory(config)
            a2a_client = factory.create(agent_card)

            message = Message(
                role=Role.ROLE_USER,
                parts=[Part(text=message_text)],
                message_id=uuid4().hex,
            )
            request = SendMessageRequest(
                message=message,
                configuration=SendMessageConfiguration(),
            )

            chunks: list[str] = []
            artifacts: list[Any] = []
            async for response in a2a_client.send_message(request):
                if response.HasField("status_update"):
                    event = response.status_update
                    if event.status.HasField("message"):
                        chunk = get_message_text(event.status.message)
                        if chunk:
                            chunks.append(chunk)
                elif response.HasField("artifact_update"):
                    artifacts.append(response.artifact_update.artifact)

            self.results["streaming"] = {
                "status": "passed",
                "detail": f"收到 {len(chunks)} 个文本片段, {len(artifacts)} 个 artifact",
                "data": {"chunks": chunks, "artifacts": artifacts},
            }
        except Exception as exc:  # noqa: BLE001
            self.results["streaming"] = {
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "data": None,
            }
