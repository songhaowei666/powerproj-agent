"""LangGraph 状态图定义与节点实现。"""

import base64
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from planning_agent.database import ProjectDatabase
from planning_agent.file_manager import FileManager
from planning_agent.models import PlanningState
from planning_agent.project_matcher import ProjectMatcher


# ---------- Prompts ----------

INTENT_SYSTEM_PROMPT = """你是一位智能助手，负责解析用户的自然语言意图。

请判断用户的操作意图，只能从以下类型中选择：
- query_project：查询项目信息（包括明细查询和聚合统计查询，如"查一下XX项目"、"所有项目变电容量的总和"）
- upload_file：上传文件到指定节点
- download_file：下载/查看指定节点的文件
- delete_file：删除指定文件
- unknown：意图不明

输出要求：仅输出意图类型字符串，不要任何解释。"""

AGGREGATE_SYSTEM_PROMPT = """你是一位 SQL 专家，负责将用户的自然语言聚合查询转换为安全的 SQLite SQL 语句。

数据库表结构：
- project_info(id, project_name, project_code, voltage_level, unit_code, line_length, substation_capacity)

要求：
1. 仅生成 SELECT 聚合查询，禁止生成 INSERT/UPDATE/DELETE/DROP 等修改语句
2. 支持的聚合函数：SUM, AVG, COUNT, MAX, MIN
3. 如果用户没有指定 WHERE 条件，对整个表进行聚合
4. 如果用户有条件（如电压等级、单位编码），在 WHERE 中体现
5. 返回单个结果，别名用中文描述，如 SUM(substation_capacity) AS 变电容量总和
6. 仅输出 SQL 语句本身，不要任何 markdown 代码块标记或其他解释
"""

NODE_CODE_MAP = {
    "可研设计": "001",
    "可研评审": "002",
    "可研批复": "003",
    "001": "001",
    "002": "002",
    "003": "003",
}


async def _is_aggregate_query(query: str, llm: BaseChatModel) -> bool:
    """使用 LLM 判断是否为聚合查询（SUM/AVG/COUNT/MAX/MIN 等统计类查询）。

    聚合查询特征：要求对多行数据进行汇总统计，如求和、平均值、计数、
    最大值、最小值等。非聚合查询特征：查询某个具体项目的属性详情。

    LLM 判断失败时回退到关键词匹配。
    """
    try:
        structured_llm = llm.with_structured_output(
            schema=dict,
            method="json_mode",
        )
        result = await structured_llm.ainvoke(
            [
                (
                    "system",
                    "你是查询分类专家。判断用户查询是否为聚合统计查询。"
                    "聚合查询：要求汇总统计（求和/平均/计数/最大/最小）。"
                    "非聚合查询：查询具体项目属性详情。"
                    "仅输出 JSON：{\"is_aggregate\": true/false}",
                ),
                ("human", query),
            ]
        )
        return bool(result.get("is_aggregate", False))
    except Exception:
        # LLM 失败时回退到关键词匹配
        aggregate_keywords = [
            "总和", "平均", "一共", "总计", "统计", "多少", "几个",
            "sum", "avg", "count", "max", "min", "total",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in aggregate_keywords)


def _extract_node_code(query: str) -> str:
    """从 query 中提取节点编码。"""
    for key, code in NODE_CODE_MAP.items():
        if key in query:
            return code
    return ""


def _is_positive_response(text: str) -> bool:
    """判断用户是否为肯定回答。"""
    positive = ["是", "对", "正确", "确认", "是的", "没错", "ok", "yes", "y"]
    text_lower = text.lower().strip()
    return any(p in text_lower for p in positive)


def _is_negative_response(text: str) -> bool:
    """判断用户是否为否定回答。"""
    negative = ["否", "不对", "错误", "不是", "取消", "no", "n", "否定的"]
    text_lower = text.lower().strip()
    return any(n in text_lower for n in negative)


# ---------- Graph Builder ----------

def build_planning_graph(
    llm: BaseChatModel,
    db: ProjectDatabase,
    fm: FileManager,
):
    """构建并编译 Planning Agent LangGraph。"""

    matcher = ProjectMatcher(db, llm)

    async def parse_intent(state: PlanningState) -> PlanningState:
        """解析用户意图。"""
        structured_llm = llm.with_structured_output(
            schema=dict,
            method="json_mode",
        )
        try:
            result = await structured_llm.ainvoke(
                [
                    ("system", INTENT_SYSTEM_PROMPT),
                    (
                        "human",
                        f'请解析以下用户输入的意图，返回 JSON {{"intent": "意图类型"}}：\n{state.query}',
                    ),
                ]
            )
            intent = result.get("intent", "unknown")
            if intent not in [
                "query_project",
                "upload_file",
                "download_file",
                "delete_file",
            ]:
                intent = "unknown"
            state.intent = intent
        except Exception:
            state.intent = "unknown"
        return state

    async def match_project(state: PlanningState) -> PlanningState:
        """匹配项目。聚合查询跳过项目匹配。"""
        if state.intent == "unknown":
            state.status = "failed"
            state.result_text = "无法理解您的意图，请明确说明您想查询项目、上传文件、下载文件还是删除文件。"
            return state

        # 聚合查询不需要匹配具体项目
        if state.intent == "query_project" and await _is_aggregate_query(state.query, llm):
            return state

        # 文件操作也需要先匹配项目
        matched = await matcher.match(state.query)
        if matched is None:
            state.status = "failed"
            state.result_text = "未找到匹配的项目，请检查项目名称或提供更多关键词。"
            return state

        state.matched_project = matched
        return state

    async def confirm_project(state: PlanningState) -> PlanningState:
        """确认项目。聚合查询跳过确认。"""
        if state.status == "failed":
            return state

        # 聚合查询不需要确认项目
        if state.intent == "query_project" and await _is_aggregate_query(state.query, llm):
            state.project_confirmed = True
            return state

        if state.project_confirmed or state.matched_project is None:
            return state

        proj = state.matched_project
        question = (
            f"找到最匹配的项目：\n"
            f"名称：{proj.project_name}\n"
            f"编码：{proj.project_code}\n"
            f"电压等级：{proj.voltage_level or '未填写'}\n"
            f"单位编码：{proj.unit_code or '未填写'}\n"
            f"线路长度：{proj.line_length}km\n"
            f"变电容量：{proj.substation_capacity}MVA\n\n"
            f"请问是这个项目吗？请回复'是'或'否'。"
        )
        user_reply = interrupt({"question": question})
        reply_text = str(user_reply)

        # 优先判断否定，避免"不是"被"是"误匹配
        if _is_negative_response(reply_text):
            state.matched_project = None
            state.status = "failed"
            state.result_text = "项目匹配失败，请重新描述您要操作的项目。"
        elif _is_positive_response(reply_text):
            state.project_confirmed = True
        else:
            # 模糊回答，视为需要再次确认
            state.project_confirmed = False

        return state

    def resolve_params(state: PlanningState) -> PlanningState:
        """解析附加参数。"""
        if state.status == "failed":
            return state

        # 解析节点编码
        node_code = _extract_node_code(state.query)
        if node_code:
            state.node_code = node_code

        # 解析待上传文件（从 state.pending_files 中，由 server 注入）
        # pending_files 已在 server 中解析并放入 state

        # 解析目标文件名（用于删除）
        if state.intent in ("download_file", "delete_file"):
            # 简单提取文件名：找引号内或最后几个字符
            # 这里简化处理，实际可由 LLM 提取
            pass

        return state

    async def execute_action(state: PlanningState) -> PlanningState:
        """执行具体操作。"""
        if state.status == "failed":
            return state

        intent = state.intent

        if intent == "query_project":
            if await _is_aggregate_query(state.query, llm):
                # 聚合查询
                sql = await _generate_aggregate_sql(llm, state.query)
                result = db.execute_aggregate_query(sql)
                state.result_text = _format_aggregate_result(result)
            else:
                # 明细查询
                if state.matched_project is None:
                    state.status = "failed"
                    state.result_text = "未匹配到项目，无法查询详情。"
                    return state
                result = db.get_project_by_code(
                    state.matched_project.project_code
                )
                if result:
                    state.result_text = _format_project_info(result)
                else:
                    state.status = "failed"
                    state.result_text = "未找到项目详细信息。"

        elif intent == "upload_file":
            if state.matched_project is None:
                state.status = "failed"
                state.result_text = "未匹配到项目，无法上传文件。"
                return state
            if not state.node_code:
                state.status = "failed"
                state.result_text = "未指定节点编码（001可研设计/002可研评审/003可研批复），无法上传文件。"
                return state
            if not state.pending_files:
                state.status = "failed"
                state.result_text = "未检测到上传的文件内容。"
                return state

            uploaded = []
            for f in state.pending_files:
                content = f["content"]
                if isinstance(content, str):
                    content = base64.b64decode(content)
                file_id = fm.save_uploaded_file(
                    project_code=state.matched_project.project_code,
                    node_code=state.node_code,
                    file_name=f["name"],
                    content_bytes=content,
                )
                db.add_file_record(
                    project_code=state.matched_project.project_code,
                    node_code=state.node_code,
                    file_id=file_id,
                    file_name=f["name"],
                    file_path=str(
                        fm.base_dir
                        / state.matched_project.project_code
                        / state.node_code
                        / f["name"]
                    ),
                )
                uploaded.append(f["name"])

            node_name = _get_node_name(state.node_code)
            state.result_text = (
                f"成功上传 {len(uploaded)} 个文件到「{node_name}」节点：\n"
                + "\n".join(f"- {n}" for n in uploaded)
            )

        elif intent == "download_file":
            if state.matched_project is None:
                state.status = "failed"
                state.result_text = "未匹配到项目，无法查询文件。"
                return state

            files = db.list_files(
                state.matched_project.project_code, state.node_code
            )
            if not files:
                node_desc = f"「{_get_node_name(state.node_code)}」节点" if state.node_code else "所有节点"
                state.result_text = f"该项目在{node_desc}下暂无文件。"
                return state

            artifacts: List[Dict[str, Any]] = [
                {"type": "text", "text": "为您找到以下文件："}
            ]
            for f in files:
                node_name = _get_node_name(f["node_code"])
                artifacts.append(
                    {
                        "type": "text",
                        "text": f"- {f['file_name']} ({node_name}) - 上传时间：{f['created_at']}",
                    }
                )
                artifacts.append(
                    {
                        "type": "file",
                        "file": {
                            "uri": fm.build_download_url(
                                f["file_id"], "http://localhost:8001"
                            ),
                            "name": f["file_name"],
                        },
                    }
                )
            state.artifacts = artifacts
            state.result_text = f"共找到 {len(files)} 个文件。"

        elif intent == "delete_file":
            if state.matched_project is None:
                state.status = "failed"
                state.result_text = "未匹配到项目，无法删除文件。"
                return state

            # 二次确认
            if not state.delete_confirmed:
                file_name = state.target_file_name or "该文件"
                question = f"确认删除文件「{file_name}」吗？该操作不可恢复，请回复'确认删除'或'取消'。"
                user_reply = interrupt({"question": question})
                if "确认删除" in str(user_reply):
                    state.delete_confirmed = True
                else:
                    state.status = "completed"
                    state.result_text = "已取消删除操作。"
                    return state

            # 执行删除
            if state.target_file_id:
                file_info = db.get_file_by_id(state.target_file_id)
            elif state.target_file_name and state.node_code:
                file_info = db.get_file_by_name(
                    state.matched_project.project_code,
                    state.node_code,
                    state.target_file_name,
                )
            else:
                state.status = "failed"
                state.result_text = "未指定要删除的文件，请提供文件名。"
                return state

            if file_info:
                db.delete_file_record(file_info["file_id"])
                fm.delete_file(file_info["file_path"])
                state.result_text = f"文件「{file_info['file_name']}」已删除。"
            else:
                state.status = "failed"
                state.result_text = "未找到要删除的文件。"

        else:
            state.status = "failed"
            state.result_text = "不支持的操作类型。"

        return state

    def finalize(state: PlanningState) -> PlanningState:
        """组装最终响应。"""
        if state.status == "failed":
            state.artifacts = [
                {"type": "text", "text": state.result_text or "操作失败"}
            ]
            return state

        if state.status == "completed":
            return state

        # 组装 artifacts
        final_artifacts: List[Dict[str, Any]] = []
        if state.result_text:
            final_artifacts.append({"type": "text", "text": state.result_text})
        final_artifacts.extend(state.artifacts)

        if not final_artifacts:
            final_artifacts.append(
                {"type": "text", "text": "操作已完成，无额外信息。"}
            )

        state.artifacts = final_artifacts
        state.status = "completed"
        return state

    # ---------- 构建图 ----------
    workflow = StateGraph(PlanningState)

    workflow.add_node("parse_intent", parse_intent)
    workflow.add_node("match_project", match_project)
    workflow.add_node("confirm_project", confirm_project)
    workflow.add_node("resolve_params", resolve_params)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "match_project")
    workflow.add_edge("match_project", "confirm_project")
    workflow.add_edge("confirm_project", "resolve_params")
    workflow.add_edge("resolve_params", "execute_action")
    workflow.add_edge("execute_action", "finalize")
    workflow.add_edge("finalize", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ---------- Helpers ----------

async def _generate_aggregate_sql(llm: BaseChatModel, query: str) -> str:
    """生成聚合查询 SQL。"""
    response = await llm.ainvoke(
        [
            ("system", AGGREGATE_SYSTEM_PROMPT),
            ("human", query),
        ]
    )
    sql = str(response.content).strip()
    # 安全校验：只允许 SELECT 语句
    sql_upper = sql.upper()
    if not sql_upper.startswith("SELECT"):
        raise ValueError("生成的 SQL 不是 SELECT 语句")
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    for word in forbidden:
        if word in sql_upper:
            raise ValueError(f"生成的 SQL 包含危险关键字: {word}")
    return sql


def _format_project_info(project: Dict[str, Any]) -> str:
    return (
        f"【项目信息】\n"
        f"名称：{project.get('project_name')}\n"
        f"编码：{project.get('project_code')}\n"
        f"电压等级：{project.get('voltage_level', '未填写')}\n"
        f"单位编码：{project.get('unit_code', '未填写')}\n"
        f"线路长度：{project.get('line_length')}km\n"
        f"变电容量：{project.get('substation_capacity')}MVA"
    )


def _format_aggregate_result(result: Dict[str, Any]) -> str:
    lines = ["【聚合查询结果】"]
    for key, value in result.items():
        lines.append(f"{key}：{value}")
    return "\n".join(lines)


def _get_node_name(code: str) -> str:
    names = {"001": "可研设计", "002": "可研评审", "003": "可研批复"}
    return names.get(code, code)
