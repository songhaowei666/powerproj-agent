import asyncio  
import httpx  
from uuid import uuid4  
from a2a.client import (  
    A2ACardResolver, ClientConfig, ClientFactory  
)  
from a2a.types import Message, Role, Part, SendMessageRequest, SendMessageConfiguration  
from a2a.helpers import get_message_text
  
  
async def main() -> None:  
    async with httpx.AsyncClient(timeout=600) as httpx_client:  
        # 解析Agent Card  
        resolver = A2ACardResolver(  
            httpx_client=httpx_client,  
            base_url='http://localhost:9999'  
        )  
  
        agent_card = await resolver.get_agent_card()  
        # 配置客户端, 启用流式传输  
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
  
        # 流式接收响应  
        async for response in client.send_message(request):  
            if response.HasField('message'):
                print(get_message_text(response.message), end='', flush=True)
            elif response.HasField('status_update'):
                print(f"\n[状态: {response.status_update.status.state}]")
        print()  
  
  
if __name__ == '__main__':  
    asyncio.run(main())  