from fastapi import APIRouter

from app.graph_memory.routes import router as _graph_memory_router

from .memory_routes import router as _memory_router
from .orchestrator import router as _chat_router
from .session_routes import router as _session_router
from .tool_routes import router as _tool_router

router = APIRouter()
router.include_router(_chat_router)
router.include_router(_memory_router)
router.include_router(_session_router)
router.include_router(_tool_router)
router.include_router(_graph_memory_router)

__all__ = ["router"]
