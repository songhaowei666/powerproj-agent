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
from .video_download_agent import VideoDownloadAgent


class VideoDownloadAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = VideoDownloadAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        if not task:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        async for item in self.agent.invoke(query):
            is_task_complete = item.get('is_task_complete')
            if not is_task_complete:
                updates_text = str(item.get('updates', ''))
                agent_update_message = new_text_message(
                    text=updates_text,
                    context_id=task.context_id,
                    task_id=task.id,
                )
                await updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    message=agent_update_message,
                )
            else:
                final_message_text = item.get('final_message_text', '')
                final_message_obj = new_text_message(
                    text=final_message_text,
                    context_id=task.context_id,
                    task_id=task.id,
                )

                file_data = item.get('file_part_data', {})
                artifact_name = item.get('artifact_name', '')

                await updater.add_artifact([
                    Part(
                        url=file_data.get('uri', ''),
                        media_type=file_data.get('mimeType', 'video/mp4'),
                        filename=artifact_name,
                    )
                ])
                await updater.complete(final_message_obj)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")


skill = AgentSkill(
    id='video-download-skill',
    name='视频下载服务',
    description='提供下载视频的服务',
    tags=['video', 'download', '工具类'],
    examples=['https://gitclone.com/download1/aliendao/qwq.mp4'],
)

public_agent_card = AgentCard(
    name='视频下载服务智能体',
    description='视频下载服务智能体，提供视频下载服务',
    version='1.0.0',
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
    supported_interfaces=[AgentInterface(protocol_binding='JSONRPC', url='http://localhost:9999/')],
)


if __name__ == "__main__":
    from examples.default_server import create_server
    create_server(
        agent_card=public_agent_card,
        agent_executor=VideoDownloadAgentExecutor(),
        port=9999,
    )
