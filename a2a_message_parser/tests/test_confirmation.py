"""confirmation Part 单元测试。"""

from a2a_message_parser.confirmation import (
    CONFIRMATION_MEDIA_TYPE,
    build_confirmation_parts,
    parse_confirmation_from_parts,
)


class TestConfirmationParts:
    """测试 confirmation parts 构建与解析。"""

    def test_build_confirmation_parts_contains_text_and_data(self):
        parts = build_confirmation_parts(
            text="请问是这个项目吗？",
            action="project_confirm",
            title="请确认项目",
        )
        assert len(parts) == 2
        assert parts[0]["text"] == "请问是这个项目吗？"
        assert parts[1]["mediaType"] == CONFIRMATION_MEDIA_TYPE
        assert parts[1]["data"]["type"] == "confirmation"
        assert parts[1]["data"]["action"] == "project_confirm"
        assert len(parts[1]["data"]["options"]) == 2

    def test_parse_confirmation_from_parts(self):
        parts = build_confirmation_parts(
            text="确认删除？",
            action="delete_confirm",
        )
        parsed = parse_confirmation_from_parts(parts)
        assert parsed is not None
        assert parsed["action"] == "delete_confirm"

    def test_parse_confirmation_returns_none_for_plain_text(self):
        assert parse_confirmation_from_parts([{"text": "普通文本"}]) is None
