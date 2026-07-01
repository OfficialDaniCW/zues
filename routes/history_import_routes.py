"""
history_import_routes.py

Admin-only upload endpoint for one-time ChatGPT / Claude.ai conversation
export import — see src/history_import.py for the parser and RAG ingestion.
"""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from core.middleware import require_admin
from src.auth_helpers import get_current_user
from src.history_import import HistoryImportError, ingest_conversations, parse_export

logger = logging.getLogger(__name__)


def setup_history_import_routes() -> APIRouter:
    router = APIRouter(prefix="/api/history-import", tags=["history_import"])

    @router.post("/upload")
    async def upload_export(request: Request, file: UploadFile = File(...)):
        """Accepts a ChatGPT or Claude.ai "export data" download — either the
        raw conversations.json or the whole .zip — and ingests it into the
        shared RAG store so it's retrievable as context in future chats."""
        require_admin(request)
        owner = get_current_user(request)

        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Uploaded file is empty")

        try:
            conversations = parse_export(raw)
            if not conversations:
                raise HistoryImportError("No conversations with text content found in the export")
            summary = ingest_conversations(conversations, owner=owner)
        except HistoryImportError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            logger.error("history import failed: %s", e, exc_info=True)
            raise HTTPException(500, "Import failed") from e

        return {"ok": True, **summary}

    return router
