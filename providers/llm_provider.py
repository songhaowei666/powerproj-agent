"""统一模型实例化管理，从 config.settings 读取参数。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


if __name__ == "__main__":
    from planning_agent.project_matcher import ProjectMatchResult

    match_llm = get_llm().with_structured_output(ProjectMatchResult).invoke("北京房山电网项目")
    # match_llm = get_llm().invoke("你随便输出")

    print(match_llm)
