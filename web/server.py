"""Web 前端 FastAPI BFF 服务入口。

运行方式::

    uvicorn web.server:app --reload --port 8501

开发时配合前端::

    cd web/frontend && npm run dev
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.api.routes import router

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
HAS_FRONTEND_BUILD = (FRONTEND_DIST / "index.html").exists()

app = FastAPI(title="电网智能助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """BFF 健康检查。"""
    return {"status": "ok"}


if HAS_FRONTEND_BUILD:
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """生产环境 SPA 回退。"""
        target = FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="0.0.0.0", port=8501, reload=True)
