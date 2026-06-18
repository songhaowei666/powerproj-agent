"""计划确认 Part 构建测试。"""

from a2a_message_parser.plan import (
    PLAN_MEDIA_TYPE,
    build_plan_confirm_data,
    build_plan_confirm_parts,
    parse_plan_confirm_from_parts,
)


def test_build_plan_confirm_parts():
    body = {
        "revision": 1,
        "goal": "统计收益",
        "plan_status": "draft",
        "tasks": [
            {
                "id": "task_1",
                "name": "统计分析",
                "required_agent": "statistics-agent",
                "status": "pending",
            }
        ],
    }
    parts = build_plan_confirm_parts("请确认计划", body)
    assert parts[0]["text"] == "请确认计划"
    assert parts[1]["mediaType"] == PLAN_MEDIA_TYPE
    assert parts[1]["data"]["type"] == "plan_confirm"
    assert parts[1]["data"]["body"]["goal"] == "统计收益"


def test_parse_plan_confirm_from_parts():
    parts = build_plan_confirm_parts(
        "请确认",
        {"revision": 1, "goal": "测试", "tasks": []},
    )
    parsed = parse_plan_confirm_from_parts(parts)
    assert parsed is not None
    assert parsed["type"] == "plan_confirm"
    assert parsed["action"] == "plan_confirm"
