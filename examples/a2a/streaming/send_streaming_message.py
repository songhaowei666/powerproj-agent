import uvicorn

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill,
    TaskState, Part
)
from a2a.helpers import new_text_message, new_task_from_user_message
from ...chat_agent import ChatAgent

class TaskChatAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = ChatAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 获取用户输入，默认为 "nothing"
        query = context.get_user_input() or "nothing"
        task = context.current_task
        
        # 首次调用时创建并初始化任务
        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            
        # 初始化状态更新器
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        full_content = ""
        
        # 迭代处理智能体生成的流式文本分块
        async for chunk in self.agent.invoke(query):
            print(chunk, end="", flush=True)
            full_content += chunk
            agent_update_message = new_text_message(
                text=chunk, context_id=task.context_id, task_id=task.id
            )
            # 更新状态并实时推送消息分块到队列
            await updater.update_status(
                TaskState.TASK_STATE_WORKING, message=agent_update_message
            )
        
        # 保存最终生成的完整文本成果
        await updater.add_artifact(
            [Part(text=full_content)]
        )
        # 标记任务生命周期结束
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")





skill = AgentSkill(
    id='task-chat-skill',
    name='任务式Chat服务',
    description='提供任务式Chat服务',
    tags=['Chat', 'Task', '工具类'],
    examples=['你好，我能问个问题吗？ ', 'A2A中能使用SSE吗？ '],
)

public_agent_card = AgentCard(
    name='任务式Chat服务智能体',
    description='任务式Chat服务智能体，提供对话服务',
    # url='http://localhost:9999/',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
    supported_interfaces=[AgentInterface(protocol_binding='JSONRPC', url='http://localhost:9999/')],

)



if __name__ == "__main__":
    from examples.default_server import create_server
    create_server(agent_card=public_agent_card,agent_executor=TaskChatAgentExecutor(),port=9999)
