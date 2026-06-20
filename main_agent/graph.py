"""Main Agent LangGraph 状态图定义。"""

import asyncio
from collections import deque
from typing import List, Dict, Any, Awaitable, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from config.settings import settings
from intent_agent.agent import IntentAgent
from intent_agent.models import IntentResult, SubTask
from main_agent.agent_network import AgentNetwork
from main_agent.models import MainState, TaskOutput, InvocationTraceEntry
from main_agent.task_manager import (
    TaskManager,
    parse_plan_confirm_action,
    parse_plan_approve_selection,
    extract_plan_modify_text,
    PLAN_CONFIRM_QUESTION,
)
from main_agent.executor import (
    call_business_agent,
    build_task_parts,
    extract_artifact_text,
    _artifact_to_message_parts,
)


# ---------- Prompt 模板 ----------

SUMMARIZE_SYSTEM_PROMPT = """你是一位智能助手，负责汇总多个业务 Agent 的执行结果，向用户输出一段清晰、连贯的总结。

请根据以下各任务的执行结果，生成一段自然语言总结：
- 概括每个子任务的核心结论
- 保持简洁，突出重点
- 使用第一人称"我"来表述

输出要求：
1. 先用一段话总体概括
2. 然后分点说明各任务的关键结论，必须包含具体数值、名称等原始数据
3. 不要编造数据中不存在的信息
4. 不要泛泛而谈，必须引用各任务执行结果中的实际内容
"""

DIRECT_REPLY_SYSTEM_PROMPT = """你是一位电网智能助手，负责与用户进行友好、简洁的对话。

当用户发送问候、闲聊或与具体电网业务无关的内容时，请自然、礼貌地回复：
- 语气友好、专业，使用第一人称"我"
- 回复简洁，1~3 段即可
- 可简要介绍你能提供的业务能力（如项目信息查询、统计分析、规划、投资评估等），引导用户提出具体业务需求
- 不要编造业务数据，不要假装已经执行了查询或分析任务
"""

DEFAULT_CLARIFICATION_VAGUE = (
    "您的请求不够明确，请补充更多细节，"
    "例如您想查询、统计、规划还是投资？具体涉及哪些时间范围或项目？"
)


def _collect_registered_agents(agent_cards: List[Any]) -> set[str]:
    """从 AgentCard 列表中提取已注册的业务 Agent 名称。"""
    agents: set[str] = set[str]()
    for card in agent_cards:
        name = str(getattr(card, "name", "")).strip()
        if isinstance(card, dict):
            name = str(card.get("name", "")).strip()
        if name:
            agents.add(name)
    return agents


def _serialize_agent_cards_summary(agent_cards: List[Any]) -> List[Dict[str, Any]]:
    """将 AgentCard 序列化为轨迹展示用的简要信息。"""
    summary: List[Dict[str, Any]] = []
    for card in agent_cards:
        skills: List[Dict[str, Any]] = []
        for skill in getattr(card, "skills", []):
            if isinstance(skill, dict):
                skills.append(
                    {
                        "id": str(skill.get("id", "")),
                        "name": str(skill.get("name", "")),
                        "description": str(skill.get("description", "")),
                    }
                )
            else:
                skills.append(
                    {
                        "id": str(getattr(skill, "id", "")),
                        "name": str(getattr(skill, "name", "")),
                        "description": str(getattr(skill, "description", "")),
                    }
                )
        summary.append(
            {
                "name": str(getattr(card, "name", "")),
                "skills": skills,
            }
        )
    return summary


TracePublisher = Callable[[Dict[str, Any]], Awaitable[None]]
SummaryChunkPublisher = Callable[[str], Awaitable[None]]
ProgressPublisher = Callable[[Dict[str, Any]], Awaitable[None]]


def _get_trace_publisher(config: Optional[RunnableConfig]) -> Optional[TracePublisher]:
    """从 LangGraph config 中获取调用轨迹流式推送回调。"""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    publisher = configurable.get("publish_trace")
    return publisher if callable(publisher) else None


def _get_summary_publisher(
    config: Optional[RunnableConfig],
) -> Optional[SummaryChunkPublisher]:
    """从 LangGraph config 中获取总结分块流式推送回调。"""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    publisher = configurable.get("publish_summary_chunk")
    return publisher if callable(publisher) else None


def _get_progress_publisher(
    config: Optional[RunnableConfig],
) -> Optional[ProgressPublisher]:
    """从 LangGraph config 中获取任务进度流式推送回调。"""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    publisher = configurable.get("publish_progress")
    return publisher if callable(publisher) else None


async def _publish_task_progress(
    state: MainState,
    config: Optional[RunnableConfig] = None,
) -> None:
    """推送当前任务计划进度快照。"""
    if state.task_plan is None:
        return
    publisher = _get_progress_publisher(config)
    if publisher is None:
        return
    payload = TaskManager.build_progress_payload(state.task_plan)
    await publisher(payload)


async def _record_trace(
    state: MainState,
    entry: InvocationTraceEntry,
    config: Optional[RunnableConfig] = None,
) -> None:
    """追加一条调用轨迹，并在流式模式下实时推送。"""
    trace_dict = entry.model_dump()
    trace_dict["step"] = len(state.invocation_traces) + 1
    state.invocation_traces.append(trace_dict)
    publisher = _get_trace_publisher(config)
    if publisher is not None:
        await publisher(trace_dict)


def _resolve_clarification_prompt(result: IntentResult, fallback: str) -> str:
    """优先使用意图识别 LLM 输出的澄清问句，否则回退到默认模板。"""
    prompt = (result.clarification_prompt or "").strip()
    return prompt or fallback


async def _stream_llm_text(
    llm: BaseChatModel,
    messages: List[tuple[str, str]],
    summary_publisher: Optional[SummaryChunkPublisher],
) -> str:
    """流式调用 LLM 生成文本，并在有 publisher 时逐块推送。"""
    summary_text = ""
    streamed_any = False
    try:
        async for chunk in llm.astream(messages):
            chunk_text = str(getattr(chunk, "content", chunk))
            if not chunk_text:
                continue
            streamed_any = True
            summary_text += chunk_text
            if summary_publisher is not None:
                await summary_publisher(chunk_text)
    except Exception:
        summary_text = ""

    if not streamed_any:
        try:
            response = await llm.ainvoke(messages)
            summary_text = str(response.content)
            if summary_publisher is not None and summary_text:
                await summary_publisher(summary_text)
        except Exception:
            summary_text = ""
    return summary_text


def _find_invalid_agent_subtasks(
    subtasks: List[SubTask], registered_agents: set[str]
) -> List[SubTask]:
    """找出 required_agent 为空或未注册的子任务。"""
    invalid: List[SubTask] = []
    for subtask in subtasks:
        agent_name = (subtask.required_agent or "").strip()
        if not agent_name or agent_name not in registered_agents:
            invalid.append(subtask)
    return invalid


def _collect_task_output_texts(state: MainState) -> List[str]:
    """收集各业务 Agent 返回的原始文本结果。"""
    texts: List[str] = []
    for phase in state.phases:
        for tid in phase:
            output = state.task_outputs.get(tid)
            if not output:
                continue
            for art in output.artifacts:
                text = extract_artifact_text(art)
                if text:
                    texts.append(text)
    return texts


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
            lines.append(f"\n【{output.required_agent} - {desc}】")
            for art in output.artifacts:
                text = extract_artifact_text(art)
                if text:
                    lines.append(f"- {text}")

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
                for part in _artifact_to_message_parts(art):
                    url = part.get("url", "")
                    if not url:
                        continue
                    links.append(
                        {
                            "task_id": tid,
                            "required_agent": output.required_agent,
                            "description": subtask.description if subtask else tid,
                            "url": url,
                            "name": part.get("filename") or "文件",
                        }
                    )
    return links


def _filter_missing_texts_for_summarize(
    raw_texts: List[str],
    summary_text: str,
    file_urls: List[str],
) -> List[str]:
    """过滤 summarize 附录中与总结或文件链接重复的原文。"""
    # 仅为文件列表标题、且文件行已被 URL 去重时，不再单独附录
    _file_section_headers = {"为您找到以下文件："}
    filtered: List[str] = []
    for text in raw_texts:
        if not text or text in summary_text:
            continue
        if file_urls and any(url in text for url in file_urls):
            continue
        if file_urls and text.strip() in _file_section_headers:
            continue
        filtered.append(text)
    return filtered


def _build_file_links_appendix(
    summary_text: str,
    file_links: List[Dict[str, str]],
) -> str:
    """构建总结中尚未出现的文件下载链接附录。"""
    pending = [
        link
        for link in file_links
        if link.get("url") and link["url"] not in summary_text
    ]
    if not pending:
        return ""

    lines = "\n\n相关文件："
    for link in pending:
        name = link.get("name") or "文件"
        lines += f"\n- [{name}]({link['url']})"
    return lines


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

    async def recognize_and_check(
        state: MainState, config: RunnableConfig
    ) -> MainState:
        """调用意图识别，检查澄清条件，不足时 interrupt 等待用户补充。"""
        agent_cards = agent_network.get_cards()
        if not agent_cards:
            # 缓存为空时尝试重新发现
            agent_cards = await agent_network.discover()

        registered_agents = _collect_registered_agents(agent_cards)
        agent_fix_attempts = 0
        max_agent_fix_attempts = settings.main_agent_agent_fix_max_attempts

        while True:
            result = await intent_agent.recognize(state.query, agent_cards)
            state.intent_result = result
            await _record_trace(
                state,
                InvocationTraceEntry(
                    step=0,
                    agent_type="intent",
                    agent_name="意图识别 Agent",
                    input={
                        "query": state.query,
                        "available_agents": _serialize_agent_cards_summary(agent_cards),
                    },
                    output={"intent_result": result.model_dump()},
                    status="success",
                ),
                config,
            )

            if not result.is_business_query:
                return state

            if not result.subtasks:
                question = _resolve_clarification_prompt(
                    result, DEFAULT_CLARIFICATION_VAGUE
                )
                clarification = interrupt({"question": question})
                state.query += f"\n补充信息：{clarification}"
                continue

            if (result.clarification_prompt or "").strip():
                question = _resolve_clarification_prompt(
                    result, DEFAULT_CLARIFICATION_VAGUE
                )
                clarification = interrupt({"question": question})
                state.query += f"\n补充信息：{clarification}"
                continue

            invalid_agent_tasks = _find_invalid_agent_subtasks(
                result.subtasks, registered_agents
            )
            if invalid_agent_tasks:
                if agent_fix_attempts >= max_agent_fix_attempts:
                    descs = "、".join(
                        [
                            f"{t.description}（Agent={t.required_agent or '空'}）"
                            for t in invalid_agent_tasks
                        ]
                    )
                    fallback = (
                        f"无法为以下任务匹配业务 Agent：{descs}。"
                        "请补充更具体的业务意图，例如查询项目信息、统计分析或投资评估。"
                    )
                    question = _resolve_clarification_prompt(result, fallback)
                    clarification = interrupt({"question": question})
                    state.query += f"\n补充信息：{clarification}"
                    agent_fix_attempts = 0
                    continue

                invalid_desc = "、".join(
                    [
                        f"{t.id}({t.required_agent or '空'})"
                        for t in invalid_agent_tasks
                    ]
                )
                state.query += (
                    f"\n系统提示：子任务 {invalid_desc} 的 required_agent 无效，"
                    f"请从以下 Agent 名称中选取：{', '.join(sorted(registered_agents))}。"
                )
                agent_fix_attempts += 1
                continue

            break

        return state

    async def direct_reply(state: MainState, config: RunnableConfig) -> MainState:
        """非业务 query 直接由 LLM 回复，不调度业务 Agent。"""
        summary_publisher = _get_summary_publisher(config)
        messages = [
            ("system", DIRECT_REPLY_SYSTEM_PROMPT),
            ("human", state.query),
        ]
        summary_text = await _stream_llm_text(llm, messages, summary_publisher)
        if not summary_text.strip():
            summary_text = "您好，我是电网智能助手。您可以告诉我需要查询、统计、规划或投资分析哪方面的内容。"

        state.summary = summary_text
        state.final_artifacts = [{"type": "text", "text": summary_text}]
        state.status = "completed"
        return state

    def route_after_recognize(state: MainState) -> str:
        """意图识别后路由：非业务 query 直接回复，否则进入计划准备。"""
        if state.intent_result and not state.intent_result.is_business_query:
            return "direct_reply"
        return "prepare_plan"

    async def prepare_plan(
        state: MainState, config: RunnableConfig
    ) -> MainState:
        """将 IntentResult 转为 draft 态 ManagedTaskPlan。"""
        if state.intent_result is None:
            return state

        revision = 1
        if state.plan_revision_base is not None:
            revision = state.plan_revision_base + 1
            state.plan_revision_base = None

        state.task_plan = TaskManager.create_plan_from_intent(
            state.intent_result,
            revision=revision,
        )
        state.phases = []
        state.current_phase_idx = 0
        state.task_outputs = {}
        state.failed_task_id = None
        state.error_message = None
        state.status = "pending"
        state.replan_from_modify = False
        await _publish_task_progress(state, config)
        return state

    async def await_plan_approval(
        state: MainState, config: RunnableConfig
    ) -> MainState:
        """展示计划并等待用户确认、修改或取消。"""
        if state.task_plan is None:
            return state

        plan_summary = state.task_plan.to_plan_summary_dict()
        user_reply = interrupt(
            {
                "type": "plan_confirm",
                "question": PLAN_CONFIRM_QUESTION,
                "plan": plan_summary,
            }
        )
        action = parse_plan_confirm_action(str(user_reply))

        if action == "cancel":
            state.task_plan = TaskManager.cancel_plan(state.task_plan, "用户取消")
            state.status = "cancelled"
            await _publish_task_progress(state, config)
            return state

        if action == "modify":
            modify_text = extract_plan_modify_text(str(user_reply))
            if modify_text:
                state.query += f"\n修改计划：{modify_text}"
            else:
                state.query += f"\n{str(user_reply).strip()}"
            state.plan_revision_base = state.task_plan.revision
            state.task_plan = None
            state.replan_from_modify = True
            return state

        if action == "approve":
            selected_ids = parse_plan_approve_selection(str(user_reply))
            if selected_ids is not None:
                if not selected_ids:
                    state.task_plan = TaskManager.cancel_plan(
                        state.task_plan, "未选择任何子任务"
                    )
                    state.status = "cancelled"
                    await _publish_task_progress(state, config)
                    return state
                state.task_plan = TaskManager.apply_task_selection(
                    state.task_plan, selected_ids
                )
            state.task_plan = TaskManager.approve_plan(state.task_plan)
            await _publish_task_progress(state, config)
            return state

        # 未能识别操作，视为修改说明
        state.query += f"\n修改计划：{str(user_reply).strip()}"
        state.plan_revision_base = state.task_plan.revision
        state.task_plan = None
        state.replan_from_modify = True
        return state

    async def handle_cancelled(
        state: MainState, config: RunnableConfig
    ) -> MainState:
        """组装取消结果。"""
        if state.task_plan is not None and state.task_plan.plan_status != "cancelled":
            state.task_plan = TaskManager.cancel_plan(state.task_plan, "用户取消")
            await _publish_task_progress(state, config)

        cancel_message = TaskManager.build_cancel_message(state.task_plan)
        state.summary = cancel_message
        state.final_artifacts = [{"type": "text", "text": cancel_message}]
        state.status = "cancelled"
        return state

    def route_after_plan_approval(state: MainState) -> str:
        """计划确认后的路由。"""
        if state.status == "cancelled":
            return "handle_cancelled"
        if state.replan_from_modify:
            return "recognize_and_check"
        if state.task_plan and state.task_plan.plan_status == "approved":
            return "build_phases"
        return "await_plan_approval"

    def build_phases(state: MainState) -> MainState:
        """根据任务依赖关系拓扑排序并分层。"""
        if state.task_plan is None or state.task_plan.plan_status != "approved":
            return state

        state.task_plan = TaskManager.start_execution(state.task_plan)
        subtasks = list(state.intent_result.subtasks)
        if state.task_plan is not None:
            skipped_ids = {
                task.id
                for task in state.task_plan.tasks
                if task.status == "skipped"
            }
            if skipped_ids:
                subtasks = [item for item in subtasks if item.id not in skipped_ids]
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

    async def _await_business_call(
        tid: str,
        subtask: SubTask,
        agent_cards: List[Any],
        session_id: str,
        subtask_map: Dict[str, SubTask],
        task_outputs: Dict[str, TaskOutput],
        business_task_id: Optional[str] = None,
        resume_text: Optional[str] = None,
        user_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, Any]:
        """调用业务 Agent，返回 (task_id, result_or_exception)。"""
        try:
            result = await call_business_agent(
                subtask,
                agent_cards,
                session_id,
                task_outputs=task_outputs,
                subtask_map=subtask_map,
                business_task_id=business_task_id,
                resume_text=resume_text,
                user_attachments=user_attachments,
            )
            return tid, result
        except Exception as exc:
            return tid, exc

    async def _resume_business_until_done(
        tid: str,
        subtask: SubTask,
        agent_cards: List[Any],
        session_id: str,
        subtask_map: Dict[str, SubTask],
        task_outputs: Dict[str, TaskOutput],
        state: MainState,
        config: RunnableConfig,
        initial_result: Dict[str, Any],
    ) -> Any:
        """处理业务 Agent input_required，在图节点内 interrupt 并 resume。"""
        result = initial_result
        business_task_id = result.get("business_task_id")
        while result.get("status") == "input_required":
            trace_info = result.get("trace", {})
            agent_name = trace_info.get("agent_name", subtask.required_agent)
            await _record_trace(
                state,
                InvocationTraceEntry(
                    step=0,
                    agent_type="business",
                    agent_name=agent_name,
                    capability=subtask.required_agent,
                    phase=state.current_phase_idx,
                    task_id=tid,
                    input={
                        "endpoint": trace_info.get("endpoint", ""),
                        "subtask": trace_info.get("subtask", subtask.model_dump()),
                        "message_parts": trace_info.get("message_parts", []),
                    },
                    output={
                        "question": result.get("question", ""),
                        "parts": result.get("parts", []),
                    },
                    status="input_required",
                ),
                config,
            )
            question = result.get("question", "请补充信息")
            prefix = f"【{agent_name} · 待确认】\n"
            if not question.startswith("【"):
                question = f"{prefix}{question}"
            user_reply = interrupt(
                {
                    "type": "business_confirm",
                    "question": question,
                    "parts": result.get("parts", []),
                    "subtask_id": tid,
                    "business_task_id": business_task_id,
                    "capability": subtask.required_agent,
                    "agent_name": agent_name,
                }
            )
            _, result = await _await_business_call(
                tid,
                subtask,
                agent_cards,
                session_id,
                subtask_map,
                task_outputs,
                business_task_id=business_task_id,
                resume_text=str(user_reply),
            )
            if isinstance(result, Exception):
                return result
        return result

    async def execute_current_phase(
        state: MainState, config: RunnableConfig
    ) -> MainState:
        """并行执行当前 Phase 的所有任务。"""
        if state.current_phase_idx >= len(state.phases):
            return state

        if state.cancel_requested and state.task_plan is not None:
            state.task_plan = TaskManager.mark_remaining_skipped(state.task_plan)
            state.task_plan = TaskManager.cancel_plan(
                state.task_plan, "用户取消执行"
            )
            state.status = "cancelled"
            await _publish_task_progress(state, config)
            return state

        agent_cards = agent_network.get_cards()
        phase_task_ids = state.phases[state.current_phase_idx]
        subtask_map: Dict[str, SubTask] = {
            t.id: t for t in state.intent_result.subtasks
        }

        session_id = state.session_id or "default"
        phase_results: Dict[str, Any] = {}
        pending_tids = list(phase_task_ids)

        for tid in pending_tids:
            if state.task_plan is not None:
                state.task_plan = TaskManager.mark_task_in_progress(
                    state.task_plan, tid
                )
        await _publish_task_progress(state, config)

        while pending_tids:
            pending_tasks: Dict[asyncio.Task[tuple[str, Any]], str] = {}
            for tid in pending_tids:
                subtask = subtask_map.get(tid)
                if not subtask:
                    state.failed_task_id = tid
                    state.error_message = f"任务 {tid} 未找到对应的 SubTask"
                    state.status = "failed"
                    return state
                pending_tasks[
                    asyncio.create_task(
                        _await_business_call(
                            tid,
                            subtask,
                            agent_cards,
                            session_id,
                            subtask_map,
                            state.task_outputs,
                            user_attachments=state.user_attachments,
                        )
                    )
                ] = tid

            pending = set(pending_tasks.keys())
            input_required_tid: Optional[str] = None
            input_required_result: Optional[Dict[str, Any]] = None
            completed_tids: List[str] = []

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    tid, result = await task
                    if isinstance(result, Exception):
                        for pending_task in pending:
                            pending_task.cancel()
                        subtask = subtask_map[tid]
                        await _record_trace(
                            state,
                            InvocationTraceEntry(
                                step=0,
                                agent_type="business",
                                agent_name=subtask.required_agent,
                                capability=subtask.required_agent,
                                phase=state.current_phase_idx,
                                task_id=tid,
                                input={
                                    "subtask": subtask.model_dump(),
                                    "message_parts": build_task_parts(
                                        subtask,
                                        state.task_outputs,
                                        subtask_map,
                                    ),
                                },
                                output={"error": str(result)},
                                status="failed",
                            ),
                            config,
                        )
                        state.failed_task_id = tid
                        state.error_message = (
                            f"任务 {tid}（{subtask.required_agent}）"
                            f"执行失败（已重试3次）：{str(result)}"
                        )
                        state.status = "failed"
                        if state.task_plan is not None:
                            state.task_plan = TaskManager.mark_task_failed(
                                state.task_plan, tid, str(result)
                            )
                            state.task_plan = TaskManager.mark_plan_failed(
                                state.task_plan
                            )
                            await _publish_task_progress(state, config)
                        return state

                    if result.get("status") == "input_required":
                        input_required_tid = tid
                        input_required_result = result
                        for pending_task in pending:
                            pending_task.cancel()
                        pending.clear()
                        break

                    phase_results[tid] = result
                    completed_tids.append(tid)

            for tid in completed_tids:
                pending_tids.remove(tid)

            if input_required_tid is not None and input_required_result is not None:
                subtask = subtask_map[input_required_tid]
                final_result = await _resume_business_until_done(
                    input_required_tid,
                    subtask,
                    agent_cards,
                    session_id,
                    subtask_map,
                    state.task_outputs,
                    state,
                    config,
                    input_required_result,
                )
                if isinstance(final_result, Exception):
                    await _record_trace(
                        state,
                        InvocationTraceEntry(
                            step=0,
                            agent_type="business",
                            agent_name=subtask.required_agent,
                            capability=subtask.required_agent,
                            phase=state.current_phase_idx,
                            task_id=input_required_tid,
                            input={"subtask": subtask.model_dump()},
                            output={"error": str(final_result)},
                            status="failed",
                        ),
                        config,
                    )
                    state.failed_task_id = input_required_tid
                    state.error_message = (
                        f"任务 {input_required_tid}（{subtask.required_agent}）"
                        f"执行失败（已重试3次）：{str(final_result)}"
                    )
                    state.status = "failed"
                    if state.task_plan is not None:
                        state.task_plan = TaskManager.mark_task_failed(
                            state.task_plan,
                            input_required_tid,
                            str(final_result),
                        )
                        state.task_plan = TaskManager.mark_plan_failed(state.task_plan)
                        await _publish_task_progress(state, config)
                    return state

                phase_results[input_required_tid] = final_result
                pending_tids.remove(input_required_tid)
                continue

            break

        for tid in phase_task_ids:
            result = phase_results[tid]
            subtask = subtask_map[tid]
            trace_info = result.get("trace", {})
            await _record_trace(
                state,
                InvocationTraceEntry(
                    step=0,
                    agent_type="business",
                    agent_name=trace_info.get("agent_name", subtask.required_agent),
                    capability=subtask.required_agent,
                    phase=state.current_phase_idx,
                    task_id=tid,
                    input={
                        "endpoint": trace_info.get("endpoint", ""),
                        "subtask": trace_info.get("subtask", subtask.model_dump()),
                        "message_parts": trace_info.get("message_parts", []),
                        "request": trace_info.get("request", {}),
                    },
                    output={"artifacts": result.get("artifacts", [])},
                    status="success",
                ),
                config,
            )
            state.task_outputs[tid] = TaskOutput(
                task_id=tid,
                required_agent=subtask.required_agent,
                status="success",
                artifacts=result.get("artifacts", []),
            )
            if state.task_plan is not None:
                state.task_plan = TaskManager.mark_task_completed(state.task_plan, tid)

        await _publish_task_progress(state, config)
        state.current_phase_idx += 1
        return state

    def finalize(state: MainState) -> MainState:
        """组装原始结果到 final_artifacts。"""
        if state.status == "cancelled":
            if state.task_plan is not None:
                cancel_message = TaskManager.build_cancel_message(state.task_plan)
                state.summary = cancel_message
                state.final_artifacts = [{"type": "text", "text": cancel_message}]
            return state

        if state.status == "failed":
            if state.task_plan is not None:
                state.task_plan = TaskManager.mark_plan_failed(state.task_plan)
            state.final_artifacts = [
                {"type": "text", "text": f"执行失败：{state.error_message}"}
            ]
            if state.invocation_traces:
                state.final_artifacts.append(
                    {
                        "type": "invocation_trace",
                        "traces": list(state.invocation_traces),
                    }
                )
            return state

        if state.task_plan is not None:
            state.task_plan = TaskManager.mark_plan_completed(state.task_plan)

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
                            "required_agent": output.required_agent,
                            "artifacts": output.artifacts,
                        }
                    )

        state.final_artifacts = raw_artifacts
        return state

    async def summarize(state: MainState, config: RunnableConfig) -> MainState:
        """调用 LLM 生成自然语言总结。"""
        if state.status == "failed":
            state.summary = state.error_message
            return state

        if not state.final_artifacts:
            state.summary = "未产生任何执行结果。"
            return state

        user_message = _build_summarize_user_message(state)
        file_links = _extract_file_links(state)
        file_urls = [link["url"] for link in file_links if link.get("url")]
        summary_publisher = _get_summary_publisher(config)
        summary_text = await _stream_llm_text(
            llm,
            [
                ("system", SUMMARIZE_SYSTEM_PROMPT),
                ("human", user_message),
            ],
            summary_publisher,
        )
        if not summary_text.strip():
            summary_text = "任务执行完成。以下是各业务 Agent 的原始结果："

        # 确保总结包含业务 Agent 返回的具体数据（跳过已在总结或文件链接中的原文）
        raw_texts = _collect_task_output_texts(state)
        missing_texts = _filter_missing_texts_for_summarize(
            raw_texts, summary_text, file_urls
        )
        if missing_texts:
            appendix = "\n\n" + "\n\n".join(missing_texts)
            summary_text += appendix
            if summary_publisher is not None:
                await summary_publisher(appendix)

        link_lines = _build_file_links_appendix(summary_text, file_links)
        if link_lines:
            summary_text += link_lines
            if summary_publisher is not None:
                await summary_publisher(link_lines)

        state.summary = summary_text

        # 成功路径不再附加 invocation_trace artifact，轨迹已在 WORKING 阶段流式推送
        final_artifacts = [
            {"type": "text", "text": summary_text}
        ]
        final_artifacts.extend(state.final_artifacts)
        state.final_artifacts = final_artifacts

        state.status = "completed"
        return state

    def route_after_execution(state: MainState) -> str:
        """条件路由：判断是否还有更多 Phase 需要执行。"""
        if state.status == "cancelled":
            return "handle_cancelled"
        if state.status == "failed":
            return "finalize"
        if state.current_phase_idx >= len(state.phases):
            return "finalize"
        return "execute_current_phase"

    # ---------- 构建图 ----------
    workflow = StateGraph(MainState)

    workflow.add_node("recognize_and_check", recognize_and_check)
    workflow.add_node("direct_reply", direct_reply)
    workflow.add_node("prepare_plan", prepare_plan)
    workflow.add_node("await_plan_approval", await_plan_approval)
    workflow.add_node("handle_cancelled", handle_cancelled)
    workflow.add_node("build_phases", build_phases)
    workflow.add_node("execute_current_phase", execute_current_phase)
    workflow.add_node("finalize", finalize)
    workflow.add_node("summarize", summarize)

    workflow.set_entry_point("recognize_and_check")
    workflow.add_conditional_edges(
        "recognize_and_check",
        route_after_recognize,
        {
            "direct_reply": "direct_reply",
            "prepare_plan": "prepare_plan",
        },
    )
    workflow.add_edge("direct_reply", END)
    workflow.add_edge("prepare_plan", "await_plan_approval")
    workflow.add_conditional_edges(
        "await_plan_approval",
        route_after_plan_approval,
        {
            "handle_cancelled": "handle_cancelled",
            "recognize_and_check": "recognize_and_check",
            "build_phases": "build_phases",
            "await_plan_approval": "await_plan_approval",
        },
    )
    workflow.add_edge("handle_cancelled", END)
    workflow.add_edge("build_phases", "execute_current_phase")
    workflow.add_conditional_edges(
        "execute_current_phase",
        route_after_execution,
        {
            "execute_current_phase": "execute_current_phase",
            "finalize": "finalize",
            "handle_cancelled": "handle_cancelled",
        },
    )
    workflow.add_edge("finalize", "summarize")
    workflow.add_edge("summarize", END)

    memory = MemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("intent_agent.models", "IntentResult"),
                ("intent_agent.models", "SubTask"),
                ("main_agent.models", "TaskOutput"),
                ("main_agent.task_models", "ManagedTaskPlan"),
                ("main_agent.task_models", "ManagedTask"),
                ("main_agent.task_models", "ProgressEvent"),
            ]
        )
    )
    return workflow.compile(checkpointer=memory)
