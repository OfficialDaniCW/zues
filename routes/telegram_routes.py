"""
telegram_routes.py

Admin-only routes to configure and pair the Telegram bridge — see
src/telegram_bridge.py for the poller and agent-turn logic.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.middleware import require_admin
from src import telegram_bridge

logger = logging.getLogger(__name__)


class TelegramTokenRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=200)


def setup_telegram_routes() -> APIRouter:
    router = APIRouter(prefix="/api/telegram", tags=["telegram"])

    @router.get("/status")
    async def status(request: Request):
        require_admin(request)
        return telegram_bridge.get_status()

    @router.post("/config")
    async def set_config(request: Request, body: TelegramTokenRequest):
        require_admin(request)
        try:
            bot = await telegram_bridge.set_token(body.token)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            logger.warning("telegram token verification failed: %s", e)
            raise HTTPException(400, "Could not verify token with Telegram") from e
        return {"ok": True, "bot": bot}

    @router.post("/pair")
    async def pair(request: Request):
        require_admin(request)
        try:
            result = telegram_bridge.start_pairing()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {
            "ok": True,
            **result,
            "instructions": "Message your bot on Telegram now — its next message binds this chat to your account.",
        }

    @router.post("/enable")
    async def enable_route(request: Request):
        require_admin(request)
        try:
            telegram_bridge.enable()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @router.post("/disable")
    async def disable_route(request: Request):
        require_admin(request)
        telegram_bridge.disable()
        return {"ok": True}

    @router.post("/test")
    async def test(request: Request):
        require_admin(request)
        try:
            await telegram_bridge.send_test_message()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    telegram_bridge.start_poller()
    return router
