"""统一模型实例化管理，从 config.settings 读取参数。"""

from langchain_openai import ChatOpenAI
from config import settings


def get_llm() -> ChatOpenAI:
    """获取统一配置的 ChatOpenAI 实例。"""
    kwargs = {"model": settings.chat_model}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
    return ChatOpenAI(**kwargs)
