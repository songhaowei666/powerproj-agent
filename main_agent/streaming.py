"""主控 Agent 流式推送协议常量与工具函数。"""

import json
from typing import Any, Dict

# 单条调用轨迹（WORKING 阶段逐条推送）
TRACE_STEP_PREFIX = "__INVOCATION_TRACE_STEP__\n"

# LLM 总结文本分块
SUMMARY_CHUNK_PREFIX = "__SUMMARY_CHUNK__\n"

# 任务计划进度快照
TASK_PROGRESS_PREFIX = "__TASK_PROGRESS__\n"


def format_trace_step_message(trace_dict: Dict[str, Any]) -> str:
    """将单条调用轨迹格式化为流式 status 消息文本。"""
    return TRACE_STEP_PREFIX + json.dumps(trace_dict, ensure_ascii=False)


def parse_trace_step_message(text: str) -> Dict[str, Any] | None:
    """从流式 status 消息中解析单条调用轨迹。"""
    if not text.startswith(TRACE_STEP_PREFIX):
        return None
    payload = text[len(TRACE_STEP_PREFIX) :]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def format_summary_chunk_message(chunk: str) -> str:
    """将总结分块格式化为流式 status 消息文本。"""
    return SUMMARY_CHUNK_PREFIX + chunk


def parse_summary_chunk_message(text: str) -> str | None:
    """从流式 status 消息中解析总结分块。"""
    if not text.startswith(SUMMARY_CHUNK_PREFIX):
        return None
    return text[len(SUMMARY_CHUNK_PREFIX) :]


def format_task_progress_message(progress_dict: Dict[str, Any]) -> str:
    """将任务计划进度格式化为流式 status 消息文本。"""
    return TASK_PROGRESS_PREFIX + json.dumps(progress_dict, ensure_ascii=False)


def parse_task_progress_message(text: str) -> Dict[str, Any] | None:
    """从流式 status 消息中解析任务计划进度。"""
    if not text.startswith(TASK_PROGRESS_PREFIX):
        return None
    payload = text[len(TASK_PROGRESS_PREFIX) :]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
