"""
brain_viz_routes.py

Owner-scoped API for the /brain visualizer — see src/brain_viz.py for the
PCA -> t-SNE projection and cosine-similarity search.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from src.auth_helpers import get_current_user
from src.brain_viz import BrainVizError, get_points, search as brain_search

logger = logging.getLogger(__name__)


def setup_brain_viz_routes() -> APIRouter:
    router = APIRouter(prefix="/api/brain-viz", tags=["brain_viz"])

    @router.get("/points")
    async def points(request: Request, dims: int = 2, limit: int = 1500, refresh: bool = False):
        owner = get_current_user(request)
        try:
            import asyncio
            result = await asyncio.to_thread(get_points, owner, dims, limit, refresh)
        except BrainVizError as e:
            raise HTTPException(503, str(e)) from e
        except Exception as e:
            logger.error("brain-viz points failed: %s", e, exc_info=True)
            raise HTTPException(500, "Could not compute brain projection") from e
        return {"ok": True, **{k: v for k, v in result.items() if not k.startswith("_")}}

    @router.get("/search")
    async def search_route(request: Request, q: str, k: int = 15, dims: int = 2):
        owner = get_current_user(request)
        try:
            import asyncio
            result = await asyncio.to_thread(brain_search, owner, q, k, dims)
        except BrainVizError as e:
            raise HTTPException(503, str(e)) from e
        except Exception as e:
            logger.error("brain-viz search failed: %s", e, exc_info=True)
            raise HTTPException(500, "Search failed") from e
        return {"ok": True, **result}

    return router
