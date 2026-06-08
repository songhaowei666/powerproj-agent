from typing import AsyncGenerator, Dict, Any
import aiohttp
import tempfile
import asyncio
from tqdm import tqdm

# 禁用总超时，避免大文件下载中途断开
_AIO_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_read=300)


class VideoDownloadAgent:
    async def invoke(
        self,
        query: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            'updates': '开始下载视频...',
            'progress_percent': 0,
            'is_task_complete': False
        }

        async with aiohttp.ClientSession(timeout=_AIO_TIMEOUT) as session:
            async with session.get(query) as response:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.mp4'
                ) as temp_file:
                    total_size = int(
                        response.headers.get('content-length', 0)
                    )
                    downloaded = 0
                    progress_percent = 0
                    current_percent = 0
                    pbar = tqdm(total=100, unit='%', desc="下载进度")

                    async for chunk in response.content.iter_chunked(8192):
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_percent = int(
                                (downloaded / total_size) * 100
                            )
                        yield {
                            'updates': f'{progress_percent}%',
                            'progress_percent': progress_percent,
                            'is_task_complete': False
                        }
                        pbar.update(progress_percent - current_percent)
                        current_percent = progress_percent

                    pbar.close()

        yield {
            'final_message_text': '100',
            'file_part_data': {
                'uri': f'file://{temp_file.name}',
                'mimeType': 'video/mp4'
            },
            'artifact_name': temp_file.name,
            'is_task_complete': True
        }
async def main():
    agent = VideoDownloadAgent()
    video_url = "https://gitclone.com"
    async for update in agent.invoke(video_url):
        if update["is_task_complete"] :
            print(f"结果: {update['final_message_text']}")
            if 'file_part_data' in update:
                print(f"文件位置: {update['file_part_data']['uri']}")

if __name__ == "__main__":
    asyncio.run(main())
