import asyncio
import httpx
from uuid import uuid4
from tqdm import tqdm

from a2a.client import (
    A2ACardResolver, ClientConfig, ClientFactory
)
from a2a.types import (
    Message, Role, Part,
    SendMessageRequest, SendMessageConfiguration,
)
from a2a.helpers import get_message_text


async def main() -> None:
    video_url = "https://gitclone.com/download1/aliendao/qwq.mp4"

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
            parts=[Part(text=video_url)],
            message_id=uuid4().hex
        )
        request = SendMessageRequest(
            message=message,
            configuration=SendMessageConfiguration(),
        )

        # 流式接收响应
        progress_percent = 0
        current_percent = 0
        pbar = tqdm(total=100, unit='%', desc="下载进度")

        async for response in client.send_message(request):
            if response.HasField('status_update'):
                event = response.status_update
                if event.status.HasField('message'):
                    content = get_message_text(event.status.message)
                    try:
                        progress_percent = int(content.replace('%', ''))
                    except ValueError:
                        pass
                    pbar.update(progress_percent - current_percent)
                    current_percent = progress_percent

            elif response.HasField('artifact_update'):
                pbar.close()
                artifact = response.artifact_update.artifact
                for part in artifact.parts:
                    if part.HasField('url'):
                        print(f"\n下载完成！文件URI: {part.url}")
                    if part.filename:
                        print(f"文件名: {part.filename}")

        if not pbar.disable:
            pbar.close()


if __name__ == '__main__':
    asyncio.run(main())
