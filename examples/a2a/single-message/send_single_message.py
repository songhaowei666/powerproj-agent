import uvicorn
from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.helpers import new_text_message
from chat_agent import ChatAgent


skill = AgentSkill(
    id='stream-chat-skill',
    name='流式Chat服务',
    description='提供流式Chat服务',
    tags=['Chat', 'Stream', '工具类'],
    examples=['你好,我能问个问题吗?', 'A2A中能使用SSE吗?'],
)

public_agent_card = AgentCard(
    name='流式Chat服务智能体',
    description='流式Chat服务智能体,提供对话服务',
    url='http://localhost:9999/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
)


class StreamChatAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = ChatAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input() + "/nothink"
        full_context = ""
        async for chunk in self.agent.invoke(query):
            print(chunk, end="", flush=True)
            full_context = full_context + chunk
            await event_queue.enqueue_event(
                new_text_message(full_context))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')


request_handler = DefaultRequestHandler(
    agent_executor=StreamChatAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=public_agent_card,
)

agent_card_routes = create_agent_card_routes(public_agent_card)
jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url='/')

app = Starlette(routes=agent_card_routes + jsonrpc_routes)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9999)
