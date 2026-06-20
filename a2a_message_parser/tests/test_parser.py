"""a2a_message_parser 单元测试。"""

from a2a_message_parser.parser import (
    build_upstream_header,
    format_upstream_context,
    parse_parts_list,
)


class TestParsePartsList:
    """测试 parts 列表解析。"""

    def test_single_task_query(self):
        parsed = parse_parts_list([{"text": "统计 PRJ001"}])
        assert parsed.task_query == "统计 PRJ001"
        assert parsed.upstream_sections == []
        assert parsed.raw_files == []

    def test_task_with_upstream_sections(self):
        header = build_upstream_header("t1", "skill-a", "统计")
        parsed = parse_parts_list(
            [
                {"text": "基于统计结果做规划"},
                {"text": header},
                {"text": "收益 10%"},
                {"url": "http://localhost:8001/files/1", "filename": "report.pdf"},
            ]
        )
        assert parsed.task_query == "基于统计结果做规划"
        assert len(parsed.upstream_sections) == 1
        assert parsed.upstream_sections[0].header == header
        assert len(parsed.upstream_sections[0].parts) == 2
        assert parsed.upstream_sections[0].parts[0]["text"] == "收益 10%"
        assert parsed.upstream_sections[0].parts[1]["url"].endswith("/files/1")

    def test_raw_file_extraction(self):
        parsed = parse_parts_list(
            [
                {"text": "上传文件"},
                {
                    "raw": b"pdf-content",
                    "filename": "design.pdf",
                    "mediaType": "application/pdf",
                },
            ]
        )
        assert parsed.task_query == "上传文件"
        assert len(parsed.raw_files) == 1
        assert parsed.raw_files[0]["name"] == "design.pdf"
        assert parsed.raw_files[0]["content"] == b"pdf-content"

    def test_url_attachment_in_main_message(self):
        parsed = parse_parts_list(
            [
                {"text": "北京西项目上传可研设计文件"},
                {"url": "http://localhost:8001/files/abc", "filename": "design.pdf"},
            ]
        )
        assert parsed.task_query == "北京西项目上传可研设计文件"
        assert len(parsed.attachment_files) == 1
        assert parsed.attachment_files[0]["url"].endswith("/files/abc")
        assert parsed.attachment_files[0]["name"] == "design.pdf"

    def test_format_upstream_context(self):
        header = build_upstream_header("t1", "skill-a", "统计")
        parsed = parse_parts_list(
            [
                {"text": "做规划"},
                {"text": header},
                {"text": "收益 10%"},
            ]
        )
        context = format_upstream_context(parsed)
        assert header in context
        assert "收益 10%" in context
