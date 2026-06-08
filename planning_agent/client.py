import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Role, Part


def get_message_text(message: Message) -> str:
    """从 Message 的 parts 中提取所有文本内容"""
    texts = []
    for part in message.parts:
        if part.text:
            texts.append(part.text)
    return "\n".join(texts)


async def main() -> None:
    async with httpx.AsyncClient(timeout=600) as httpx_client:
        # 解析 Agent Card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="http://localhost:8001",
        )

        agent_card = await resolver.get_agent_card()
        # 配置客户端，启用流式传输
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=True,
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)

        # 创建请求消息
        request_message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text="请用100字简单介绍Python编程语言")],
            message_id=uuid4().hex,
        )

        # 流式接收响应
        async for response in client.send_message(request_message):
            print(get_message_text(response))


if __name__ == "__main__":
    asyncio.run(main())
