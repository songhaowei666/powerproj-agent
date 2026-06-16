"""A2A Message / Part 构建工具。"""

from typing import Any, Dict, List

import google.protobuf.json_format as json_format
from a2a.types import a2a_pb2
from google.protobuf.struct_pb2 import Value


def parts_dicts_to_proto(parts: List[Dict[str, Any]]) -> List[a2a_pb2.Part]:
    """将 parts 字典列表转为 protobuf Part 列表。"""
    proto_parts: List[a2a_pb2.Part] = []
    for part in parts:
        proto_part = a2a_pb2.Part()
        if part.get("text"):
            proto_part.text = part["text"]
        elif part.get("url"):
            proto_part.url = part["url"]
            proto_part.filename = part.get("filename") or part.get("name") or ""
        elif part.get("data") is not None:
            data_value = Value()
            json_format.ParseDict(part["data"], data_value)
            proto_part.data.CopyFrom(data_value)
        else:
            continue

        media_type = part.get("mediaType") or part.get("media_type") or ""
        if media_type:
            proto_part.media_type = media_type
        proto_parts.append(proto_part)
    return proto_parts


def build_agent_message_from_parts(
    parts: List[Dict[str, Any]],
    context_id: str,
    task_id: str,
) -> a2a_pb2.Message:
    """根据 parts 字典构建 Agent 侧 Message。"""
    message = a2a_pb2.Message()
    message.role = a2a_pb2.ROLE_AGENT
    message.context_id = context_id
    message.task_id = task_id
    for proto_part in parts_dicts_to_proto(parts):
        message.parts.append(proto_part)
    return message


def message_to_parts_dicts(message: Any) -> List[Dict[str, Any]]:
    """将 protobuf Message 转为 parts 字典列表（含 data）。"""
    if message is None:
        return []
    return json_format.MessageToDict(message).get("parts", [])
