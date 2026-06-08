import uvicorn
from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from a2a.helpers import new_text_message

def get_a2a_app(agent_executor:AgentExecutor,agent_card:AgentCard):
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    agent_card_routes = create_agent_card_routes(agent_card)
    jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url='/')

    app = Starlette(routes=agent_card_routes + jsonrpc_routes)
    return app

def create_server(agent_executor:AgentExecutor,agent_card:AgentCard,port:int):
    uvicorn.run(get_a2a_app(agent_executor=agent_executor,agent_card=agent_card), host='0.0.0.0', port=port)
