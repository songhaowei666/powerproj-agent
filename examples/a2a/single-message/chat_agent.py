import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from providers.llm_provider import get_llm

prompt = ChatPromptTemplate(
    [("system","你是一个专业的AI助手"),("human","{query}")]
)

llm = get_llm()

class ChatAgent:
    def __init__(self):
        self.llm = prompt | llm

    async def invoke(self , query:str="你好"):
        response = self.llm.stream({"query":query})
        for token in response:
            chunk = token.content
            if chunk:
                await asyncio.sleep(0.001)
                yield chunk


async def main():
    agent = ChatAgent()
    # 通过一句话测试agent
    query = "你好，请介绍一下你自己"
    print(f"用户: {query}")
    print("AI: ", end="", flush=True)
    async for chunk in agent.invoke(query):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    asyncio.run(main())
