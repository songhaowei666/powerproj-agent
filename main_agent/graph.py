"""Main Agent LangGraph 状态图定义。"""

import asyncio
from collections import deque
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from intent_agent.agent import IntentAgent
from intent_agent.models import SubTask
from main_agent.agent_network import AgentNetwork
from main_agent.models import MainState, TaskOutput
from main_agent.executor import call_business_agent


# ---------- Prompt 模板 ----------

SUMMARIZE_SYSTEM_PROMPT = """你是一位智能助手，负责汇总多个业务 Agent 的执行结果，向用户输出一段清晰、连贯的总结。

请根据以下各任务的执行结果，生成一段自然语言总结：
- 概括每个子任务的核心结论
- 保持简洁，突出重点
- 使用第一人称"我"来表述

输出要求：
1. 先用一段话总体概括
2. 然后分点说明各任务的关键结论
3. 不要编造数据中不存在的信息
"""


def _build_summarize_user_message(state: MainState) -> str:
    """为 summarize 节点构建用户提示内容。"""
    lines: List[str] = ["用户原始请求：", state.query, "", "各任务执行结果："]

    subtask_map = {t.id: t for t in state.intent_result.subtasks}

    for phase in state.phases:
        for tid in phase:
            output = state.task_outputs.get(tid)
            if not output:
                continue
            subtask = subtask_map.get(tid)
            desc = subtask.description if subtask else tid
            lines.append(f"\n【{output.required_capability} - {desc}】")
            for art in output.artifacts:
                if art.get("type") == "text":
                    lines.append(f"- {art.get('text', '')}")

    return "\n".join(lines)


def _extract_file_links(state: MainState) -> List[Dict[str, str]]:
    """从所有 task outputs 中提取文件下载链接。"""
    links: List[Dict[str, str]] = []
    subtask_map = {t.id: t for t in state.intent_result.subtasks}

    for phase in state.phases:
        for tid in phase:
            output = state.task_outputs.get(tid)
            if not output:
                continue
            subtask = subtask_map.get(tid)
            for art in output.artifacts:
                if art.get("type") == "file" or "url" in art:
                    links.append(
                        {
                            "task_id": tid,
                            "required_capability": output.required_capability,
                            "description": subtask.description if subtask else tid,
                            "url": art.get("url", ""),
                            "name": art.get("name", art.get("filename", "文件")),
                        }
                    )
    return links


# ---------- 节点实现 ----------


def build_main_graph(llm: BaseChatModel, agent_network: AgentNetwork):
    """构建并编译 Main Agent LangGraph。

    Args:
        llm: LangChain ChatModel 实例
        agent_network: Agent 网络管理器，用于获取可用 AgentCard

    Returns:
        CompiledStateGraph
    """
    intent_agent = IntentAgent(llm)

    async def recognize_and_check(state: MainState) -> MainState:
        """调用意图识别，检查置信度，不足时 interrupt 等待用户补充。"""
        agent_cards = agent_network.get_cards()
        if not agent_cards:
            # 缓存为空时尝试重新发现
            agent_cards = await agent_network.discover()

        while True:
            result = await intent_agent.recognize(state.query, agent_cards)
            state.intent_result = result

            if not result.subtasks:
                question = (
                    "您的请求不够明确，请补充更多细节，"
                    "例如您想查询、统计、规划还是投资？具体涉及哪些时间范围或项目？"
                )
                clarification = interrupt({"question": question})
                state.query += f"\n补充信息：{clarification}"
                continue

            low_conf_tasks = [t for t in result.subtasks if t.confidence < 0.8]
            if low_conf_tasks:
                descs = "、".join(
                    [f"{t.description}（置信度{t.confidence:.2f}）" for t in low_conf_tasks]
                )
                question = f"以下任务置信度较低，请补充相关信息：{descs}"
                clarification = interrupt({"question": question})
                state.query += f"\n补充信息：{clarification}"
                continue

            break

        return state

    def build_phases(state: MainState) -> MainState:
        """根据任务依赖关系拓扑排序并分层。"""
        subtasks = state.intent_result.subtasks
        task_ids = {t.id for t in subtasks}

        # 计算入度与依赖关系
        in_degree: Dict[str, int] = {t.id: 0 for t in subtasks}
        dependents: Dict[str, List[str]] = {t.id: [] for t in subtasks}

        for t in subtasks:
            for dep in t.dependencies:
                if dep in task_ids:
                    dependents[dep].append(t.id)
                    in_degree[t.id] += 1
                # 忽略不存在的依赖（容错）

        # Kahn 算法分层
        phases: List[List[str]] = []
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])

        while queue:
            current_phase = list(queue)
            phases.append(current_phase)
            queue = deque()
            for tid in current_phase:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        queue.append(dep_tid)

        # 检查是否有环
        scheduled = set()
        for phase in phases:
            scheduled.update(phase)

        if scheduled != task_ids:
            # 有环，退化为单 Phase
            phases = [list(task_ids)]

        state.phases = phases
        state.current_phase_idx = 0
        state.status = "executing"
        return state

    async def execute_current_phase(state: MainState) -> MainState:
        """并行执行当前 Phase 的所有任务。"""
        if state.current_phase_idx >= len(state.phases):
            return state

        agent_cards = agent_network.get_cards()
        phase_task_ids = state.phases[state.current_phase_idx]
        subtask_map: Dict[str, SubTask] = {
            t.id: t for t in state.intent_result.subtasks
        }

        session_id = state.session_id or "default"
        coros = []
        for tid in phase_task_ids:
            subtask = subtask_map.get(tid)
            if not subtask:
                state.failed_task_id = tid
                state.error_message = f"任务 {tid} 未找到对应的 SubTask"
                state.status = "failed"
                return state
            coros.append(call_business_agent(subtask, agent_cards, session_id))

        results = await asyncio.gather(*coros, return_exceptions=True)

        for tid, result in zip(phase_task_ids, results):
            subtask = subtask_map[tid]
            if isinstance(result, Exception):
                state.failed_task_id = tid
                state.error_message = (
                    f"任务 {tid}（{subtask.required_capability}）执行失败（已重试3次）：{str(result)}"
                )
                state.status = "failed"
                return state

            state.task_outputs[tid] = TaskOutput(
                task_id=tid,
                required_capability=subtask.required_capability,
                status="success",
                artifacts=result.get("artifacts", []),
            )

        state.current_phase_idx += 1
        return state

    def finalize(state: MainState) -> MainState:
        """组装原始结果到 final_artifacts。"""
        if state.status == "failed":
            state.final_artifacts = [
                {"type": "text", "text": f"执行失败：{state.error_message}"}
            ]
            return state

        # 按 Phase 顺序组织原始结果
        raw_artifacts: List[Dict[str, Any]] = []
        for phase in state.phases:
            for tid in phase:
                output = state.task_outputs.get(tid)
                if output:
                    raw_artifacts.append(
                        {
                            "type": "task_result",
                            "task_id": tid,
                            "required_capability": output.required_capability,
                            "artifacts": output.artifacts,
                        }
                    )

        state.final_artifacts = raw_artifacts
        return state

    async def summarize(state: MainState) -> MainState:
        """调用 LLM 生成自然语言总结。"""
        if state.status == "failed":
            state.summary = state.error_message
            return state

        if not state.final_artifacts:
            state.summary = "未产生任何执行结果。"
            return state

        user_message = _build_summarize_user_message(state)
        file_links = _extract_file_links(state)

        messages = [
            ("system", SUMMARIZE_SYSTEM_PROMPT),
            ("human", user_message),
        ]

        try:
            response = await llm.ainvoke(messages)
            summary_text = str(response.content)
        except Exception as e:
            # 总结失败不影响主流程，降级为简单拼接
            summary_text = "任务执行完成。以下是各业务 Agent 的原始结果："

        # 附加文件链接引用
        if file_links:
            summary_text += "\n\n相关文件："
            for link in file_links:
                summary_text += (
                    f"\n- {link['required_capability']}（{link['description']}）：{link['url']}"
                )

        state.summary = summary_text

        # 将总结作为第一个 artifact
        final_artifacts = [
            {"type": "text", "text": summary_text}
        ]
        final_artifacts.extend(state.final_artifacts)
        state.final_artifacts = final_artifacts

        state.status = "completed"
        return state

    def route_after_execution(state: MainState) -> str:
        """条件路由：判断是否还有更多 Phase 需要执行。"""
        if state.status == "failed":
            return "finalize"
        if state.current_phase_idx >= len(state.phases):
            return "finalize"
        return "execute_current_phase"

    # ---------- 构建图 ----------
    workflow = StateGraph(MainState)

    workflow.add_node("recognize_and_check", recognize_and_check)
    workflow.add_node("build_phases", build_phases)
    workflow.add_node("execute_current_phase", execute_current_phase)
    workflow.add_node("finalize", finalize)
    workflow.add_node("summarize", summarize)

    workflow.set_entry_point("recognize_and_check")
    workflow.add_edge("recognize_and_check", "build_phases")
    workflow.add_edge("build_phases", "execute_current_phase")
    workflow.add_conditional_edges(
        "execute_current_phase",
        route_after_execution,
        {
            "execute_current_phase": "execute_current_phase",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("finalize", "summarize")
    workflow.add_edge("summarize", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
