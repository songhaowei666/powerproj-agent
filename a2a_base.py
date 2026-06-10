"""
A2A 协议基础服务器实现（基于 a2a-sdk）
提供和 default_server.py 一致的快捷入口，统一使用 SDK 标准组件。
"""

from typing import Sequence

import uvicorn
from starlette.applications import Starlette
from starlette.routing import BaseRoute

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes


def get_a2a_app(
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    extra_routes: Sequence[BaseRoute] | None = None,
):
    """快捷函数：传入 AgentExecutor 和 AgentCard，直接拿到 Starlette app。

    Args:
        agent_executor: A2A Agent 执行器
        agent_card: Agent 卡片
        extra_routes: 额外路由列表（如文件下载路由 /files/{file_id}）
    """
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    agent_card_routes = create_agent_card_routes(agent_card)
    jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url='/')

    all_routes = list(agent_card_routes) + list(jsonrpc_routes)
    if extra_routes:
        all_routes.extend(extra_routes)

    app = Starlette(routes=all_routes)
    return app


def create_server(
    agent_executor: AgentExecutor,
    agent_card: AgentCard,
    port: int,
    host: str = '0.0.0.0',
    extra_routes: Sequence[BaseRoute] | None = None,
    **uvicorn_kwargs,
):
    """启动 A2A 服务

    Args:
        uvicorn_kwargs: 透传给 uvicorn.run 的额外参数（如 log_level, reload 等）
    """
    uvicorn.run(
        get_a2a_app(
            agent_executor=agent_executor,
            agent_card=agent_card,
            extra_routes=extra_routes,
        ),
        host=host,
        port=port,
        **uvicorn_kwargs,
    )
