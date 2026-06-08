"""自然语言项目匹配 - LLM + SQL 组合查询。"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from planning_agent.database import ProjectDatabase
from planning_agent.models import MatchedProject


class ProjectFilter(BaseModel):
    """从自然语言中提取的项目查询条件。"""

    keywords: Optional[str] = Field(
        default=None, description="项目名称或编码中的关键词"
    )
    voltage_level: Optional[str] = Field(
        default=None, description="电压等级，如 10kv/35kv/220kv/330kv/500kv/1000kv"
    )
    unit_code: Optional[str] = Field(
        default=None, description="单位编码，如 01-27"
    )
    min_line_length: Optional[float] = Field(
        default=None, description="线路长度下限 (km)"
    )
    max_line_length: Optional[float] = Field(
        default=None, description="线路长度上限 (km)"
    )
    min_substation_capacity: Optional[float] = Field(
        default=None, description="变电容量下限 (MVA)"
    )
    max_substation_capacity: Optional[float] = Field(
        default=None, description="变电容量上限 (MVA)"
    )


class ProjectMatchResult(BaseModel):
    """LLM 项目匹配结果。"""

    project_code: str = Field(..., description="最匹配的项目编码")
    reason: str = Field(default="", description="匹配理由")
    match_score: float = Field(default=1.0, description="匹配置信度 [0, 1]")


FILTER_SYSTEM_PROMPT = """你是一位电力项目数据查询专家，负责从用户的自然语言描述中提取精确的数据库查询条件。

请分析用户输入，提取以下可能的筛选条件：
- keywords: 项目名称或编码中的关键词（如"北京西"、"PRJ001"）
- voltage_level: 电压等级，仅提取标准值：10kv, 35kv, 220kv, 330kv, 500kv, 1000kv
- unit_code: 单位编码，如"01"、"02"等（两位字符）
- min_line_length / max_line_length: 线路长度范围（km）
- min_substation_capacity / max_substation_capacity: 变电容量范围（MVA）

注意：
1. 如果用户没有提到某个条件，该字段保持为 null
2. 电压等级提取时要标准化，如"500千伏"提取为"500kv"
3. 范围条件如"超过100公里" -> min_line_length=100, max_line_length=null
4. 如果用户query明显是聚合/统计类（如"总和"、"平均"、"一共"、"统计"），所有条件保持为 null
"""

MATCH_SYSTEM_PROMPT = """你是一位电力项目匹配专家，负责从候选项目中选择最匹配用户描述的一个。

请根据用户原始描述和候选项目列表，选择最匹配的项目，并给出匹配置信度。

置信度规则：
- 名称完全匹配或高度相关：0.9-1.0
- 名称部分匹配且电压等级/单位一致：0.7-0.9
- 只有部分字段匹配：0.5-0.7
- 勉强相关：0.3-0.5
"""


class ProjectMatcher:
    """项目匹配器。"""

    def __init__(self, db: ProjectDatabase, llm: BaseChatModel):
        self.db = db
        self.llm = llm

    async def match(self, query: str) -> Optional[MatchedProject]:
        """根据自然语言查询匹配最可能的项目。

        步骤：
        1. LLM 提取筛选条件
        2. 执行组合查询
        3. 若结果为空返回 None
        4. 若结果唯一直接返回
        5. 若多条用 LLM 选择最匹配的一条
        """
        # 步骤1：提取筛选条件
        structured_llm = self.llm.with_structured_output(ProjectFilter)
        filter_result = await structured_llm.ainvoke(
            [
                ("system", FILTER_SYSTEM_PROMPT),
                ("human", query),
            ]
        )

        # 步骤2：执行组合查询
        projects = self.db.search_projects(
            keywords=filter_result.keywords,
            voltage_level=filter_result.voltage_level,
            unit_code=filter_result.unit_code,
            min_line_length=filter_result.min_line_length,
            max_line_length=filter_result.max_line_length,
            min_capacity=filter_result.min_substation_capacity,
            max_capacity=filter_result.max_substation_capacity,
            limit=10,
        )

        if not projects:
            return None

        if len(projects) == 1:
            proj = projects[0]
            return MatchedProject(
                project_name=proj["project_name"],
                project_code=proj["project_code"],
                voltage_level=proj.get("voltage_level"),
                unit_code=proj.get("unit_code"),
                line_length=proj.get("line_length"),
                substation_capacity=proj.get("substation_capacity"),
                match_score=1.0,
            )

        # 步骤3：多条结果，用 LLM 选择最匹配的一条
        projects_text = "\n".join(
            [
                f"{i+1}. 编码:{p['project_code']} 名称:{p['project_name']} "
                f"电压:{p.get('voltage_level','')} 单位:{p.get('unit_code','')} "
                f"线路:{p.get('line_length','')}km 变电:{p.get('substation_capacity','')}MVA"
                for i, p in enumerate(projects)
            ]
        )

        match_llm = self.llm.with_structured_output(ProjectMatchResult)
        match_result = await match_llm.ainvoke(
            [
                ("system", MATCH_SYSTEM_PROMPT),
                (
                    "human",
                    f"用户描述：{query}\n\n候选项目：\n{projects_text}\n\n请选择最匹配的项目。",
                ),
            ]
        )

        selected = next(
            (p for p in projects if p["project_code"] == match_result.project_code),
            None,
        )
        if selected is None:
            selected = projects[0]
            match_result.match_score = 0.8

        return MatchedProject(
            project_name=selected["project_name"],
            project_code=selected["project_code"],
            voltage_level=selected.get("voltage_level"),
            unit_code=selected.get("unit_code"),
            line_length=selected.get("line_length"),
            substation_capacity=selected.get("substation_capacity"),
            match_score=match_result.match_score,
        )
