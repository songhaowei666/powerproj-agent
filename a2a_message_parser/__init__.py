"""A2A Message 解析模块，统一下游 Agent 对 message.parts 的消费。"""

from a2a_message_parser.parser import (
    UPSTREAM_MARKER_PREFIX,
    ParsedInput,
    UpstreamSection,
    build_upstream_header,
    format_upstream_context,
    parse_message_parts,
    parse_parts_list,
)

__all__ = [
    "UPSTREAM_MARKER_PREFIX",
    "ParsedInput",
    "UpstreamSection",
    "build_upstream_header",
    "format_upstream_context",
    "parse_message_parts",
    "parse_parts_list",
]
