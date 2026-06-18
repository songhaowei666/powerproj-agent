"""A2A Message 解析模块，统一下游 Agent 对 message.parts 的消费。"""

from a2a_message_parser.confirmation import (
    CONFIRMATION_MEDIA_TYPE,
    DEFAULT_YES_NO_OPTIONS,
    DELETE_CONFIRM_OPTIONS,
    build_confirmation_data,
    build_confirmation_parts,
    parse_confirmation_from_parts,
)
from a2a_message_parser.plan import (
    PLAN_MEDIA_TYPE,
    DEFAULT_PLAN_CONFIRM_OPTIONS,
    build_plan_confirm_data,
    build_plan_confirm_parts,
    parse_plan_confirm_from_parts,
)
from a2a_message_parser.message_builder import (
    build_agent_message_from_parts,
    message_to_parts_dicts,
    parts_dicts_to_proto,
)
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
    "CONFIRMATION_MEDIA_TYPE",
    "PLAN_MEDIA_TYPE",
    "DEFAULT_YES_NO_OPTIONS",
    "DELETE_CONFIRM_OPTIONS",
    "DEFAULT_PLAN_CONFIRM_OPTIONS",
    "ParsedInput",
    "UpstreamSection",
    "build_upstream_header",
    "build_confirmation_data",
    "build_confirmation_parts",
    "build_plan_confirm_data",
    "build_plan_confirm_parts",
    "build_agent_message_from_parts",
    "format_upstream_context",
    "message_to_parts_dicts",
    "parse_confirmation_from_parts",
    "parse_plan_confirm_from_parts",
    "parse_message_parts",
    "parse_parts_list",
    "parts_dicts_to_proto",
]
