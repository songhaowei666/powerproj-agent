"""LangGraph StateGraph 定义与节点实现。"""

from typing import TypedDict, List, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from intent_agent.models import IntentResult
from intent_agent.prompts import build_system_prompt
from intent_agent.rag_stub import retrieve_similar_examples


class IntentState(TypedDict):
    """LangGraph 状态定义。"""

    query: str
    few_shots: List[Dict]
    result: Optional[IntentResult]


async def retrieve_few_shots_node(state: IntentState) -> IntentState:
    """调用 rag_stub 获取少样本示例。"""
    few_shots = await retrieve_similar_examples(state["query"], k=3)
    return {"query": state["query"], "few_shots": few_shots, "result": None}


async def plan_tasks_node(state: IntentState, llm: BaseChatModel) -> IntentState:
    """构建 prompt，调用 LLM 生成结构化任务规划。"""
    system_prompt = build_system_prompt(state["few_shots"])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]

    structured_llm = llm.with_structured_output(IntentResult)
    result = await structured_llm.ainvoke(messages)

    return {"query": state["query"], "few_shots": state["few_shots"], "result": result}


def build_intent_graph(llm: BaseChatModel):
    """构建并编译意图识别 LangGraph。

    Args:
        llm: LangChain 聊天模型实例

    Returns:
        编译后的 StateGraph
    """

    async def _plan_wrapper(state: IntentState) -> IntentState:
        return await plan_tasks_node(state, llm)

    builder = StateGraph(IntentState)
    builder.add_node("retrieve", retrieve_few_shots_node)
    builder.add_node("plan", _plan_wrapper)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "plan")
    builder.add_edge("plan", END)
    return builder.compile()
