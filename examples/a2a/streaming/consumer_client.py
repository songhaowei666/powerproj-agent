import asyncio
import httpx
from uuid import uuid4

from a2a.client import (
    A2ACardResolver, ClientConfig, ClientFactory
)
from a2a.types import (
    Message, Role, Part,
    SendMessageRequest, SendMessageConfiguration,
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent,
    Task, TaskState
)
from a2a.helpers import get_message_text


async def task_consumer(data, card):
    """处理流式事件：打印文本片段和 artifacts。"""
    if isinstance(data, tuple):
        task, event = data
        if isinstance(event, TaskStatusUpdateEvent):
            message = task.status.message
            if message is not None:
                chunk = get_message_text(message)
                print(chunk, end="", flush=True)
        elif isinstance(event, TaskArtifactUpdateEvent):
            print("\n\n*** artifacts ***")
            for part in event.artifact.parts:
                if part.HasField('text'):
                    print(part.text)


async def main() -> None:
    async with httpx.AsyncClient(timeout=600) as httpx_client:
        # 解析 Agent Card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url='http://localhost:9999'
        )
        agent_card = await resolver.get_agent_card()

        # 配置客户端，启用流式传输
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=True
        )

        factory = ClientFactory(config)
        client = factory.create(agent_card)

        # 创建请求消息
        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text='请用100字简单介绍Python编程语言')],
            message_id=uuid4().hex
        )
        request = SendMessageRequest(
            message=message,
            configuration=SendMessageConfiguration(),
        )

        # 流式接收响应，构建 task 对象并调用消费者
        task = None
        async for response in client.send_message(request):
            if response.HasField('status_update'):
                event = response.status_update
                if task is None:
                    task = Task(id=event.task_id, context_id=event.context_id)
                task.status.CopyFrom(event.status)
                await task_consumer((task, event), agent_card)

            elif response.HasField('artifact_update'):
                event = response.artifact_update
                if task is None:
                    task = Task(id=event.task_id, context_id=event.context_id)
                task.artifacts.append(event.artifact)
                await task_consumer((task, event), agent_card)

        print()


if __name__ == '__main__':
    asyncio.run(main())
