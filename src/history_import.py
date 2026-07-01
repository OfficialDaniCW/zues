"""
history_import.py

Imports ChatGPT and Claude.ai conversation exports into Zeus's shared RAG
store, so past conversations from those platforms become retrievable
context in future chats — the same "brain" that already grows from Zeus's
own chat history via memory extraction.

Neither ChatGPT nor Claude.ai exposes a live API for personal web-chat
history; the only way in is their "export data" download (a .zip, or the
`conversations.json` from inside it) from account settings. This module
parses both formats and pushes them in as chunked RAG documents, owner-
scoped exactly like personal documents.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 2000
MAX_FILE_BYTES = 200 * 1024 * 1024  # sanity cap on the uploaded export


class HistoryImportError(ValueError):
    pass


@dataclass
class ImportedMessage:
    role: str
    text: str
    ts: Optional[float] = None


@dataclass
class ImportedConversation:
    source: str  # "chatgpt" | "claude"
    external_id: str
    title: str
    created_at: Optional[float]
    messages: List[ImportedMessage] = field(default_factory=list)


def _find_conversations_json(raw: bytes) -> Any:
    """Accept either a raw conversations.json or the export .zip."""
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                candidates = [n for n in zf.namelist() if n.endswith("conversations.json")]
                if not candidates:
                    raise HistoryImportError(
                        "No conversations.json found inside the uploaded export .zip"
                    )
                name = min(candidates, key=len)  # prefer the top-level one
                with zf.open(name) as f:
                    return json.load(f)
        except zipfile.BadZipFile as e:
            raise HistoryImportError("Uploaded file is not a valid .zip") from e
        except json.JSONDecodeError as e:
            raise HistoryImportError("conversations.json inside the export is not valid JSON") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HistoryImportError(
            "File is not a recognizable conversations.json or export .zip"
        ) from e


def _is_chatgpt_shape(data: Any) -> bool:
    return isinstance(data, list) and any(isinstance(c, dict) and "mapping" in c for c in data[:5])


def _is_claude_shape(data: Any) -> bool:
    return isinstance(data, list) and any(isinstance(c, dict) and "chat_messages" in c for c in data[:5])


def _text_from_content_parts(parts: Any) -> str:
    out = []
    for p in parts or []:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            t = p.get("text") or p.get("content") or ""
            if isinstance(t, str) and t:
                out.append(t)
    return "\n".join(out).strip()


def parse_chatgpt_export(data: List[Dict]) -> List[ImportedConversation]:
    out: List[ImportedConversation] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping") or {}
        nodes = [n for n in mapping.values() if isinstance(n, dict) and n.get("message")]
        nodes.sort(key=lambda n: (n["message"].get("create_time") or 0))
        messages: List[ImportedMessage] = []
        for node in nodes:
            m = node["message"]
            role = (m.get("author") or {}).get("role") or ""
            if role not in ("user", "assistant"):
                continue
            text = _text_from_content_parts((m.get("content") or {}).get("parts"))
            if not text:
                continue
            messages.append(ImportedMessage(role=role, text=text, ts=m.get("create_time")))
        if messages:
            out.append(ImportedConversation(
                source="chatgpt",
                external_id=str(conv.get("id") or conv.get("conversation_id") or len(out)),
                title=str(conv.get("title") or "Untitled")[:200],
                created_at=conv.get("create_time"),
                messages=messages,
            ))
    return out


def parse_claude_export(data: List[Dict]) -> List[ImportedConversation]:
    out: List[ImportedConversation] = []
    for conv in data:
        if not isinstance(conv, dict):
            continue
        messages: List[ImportedMessage] = []
        for m in conv.get("chat_messages") or []:
            if not isinstance(m, dict):
                continue
            sender = m.get("sender") or ""
            role = "user" if sender == "human" else "assistant" if sender == "assistant" else ""
            if not role:
                continue
            text = m.get("text") or ""
            if not text and isinstance(m.get("content"), list):
                text = "\n".join(
                    c.get("text", "") for c in m["content"]
                    if isinstance(c, dict) and c.get("text")
                )
            text = (text or "").strip()
            if not text:
                continue
            messages.append(ImportedMessage(role=role, text=text, ts=m.get("created_at")))
        if messages:
            out.append(ImportedConversation(
                source="claude",
                external_id=str(conv.get("uuid") or len(out)),
                title=str(conv.get("name") or "Untitled")[:200],
                created_at=conv.get("created_at"),
                messages=messages,
            ))
    return out


def parse_export(raw: bytes) -> List[ImportedConversation]:
    if len(raw) > MAX_FILE_BYTES:
        raise HistoryImportError("Export file is too large")
    data = _find_conversations_json(raw)
    if _is_chatgpt_shape(data):
        return parse_chatgpt_export(data)
    if _is_claude_shape(data):
        return parse_claude_export(data)
    raise HistoryImportError(
        "Unrecognized export format — expected a ChatGPT or Claude.ai conversations.json"
    )


def _chunk_conversation(conv: ImportedConversation) -> List[str]:
    """Join a conversation's turns into ~MAX_CHUNK_CHARS blocks so each RAG
    document stays a retrievable size instead of one giant blob per chat."""
    lines = [f"{m.role}: {m.text}" for m in conv.messages]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for line in lines:
        if buf and buf_len + len(line) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def ingest_conversations(conversations: List[ImportedConversation], owner: Optional[str]) -> Dict[str, int]:
    """Push parsed conversations into the shared RAG store as owner-scoped
    documents, tagged with source/title so they surface in normal chat
    retrieval (`VectorRAG.search` filters only on the `owner` metadata key,
    same as personal documents)."""
    from src.rag_singleton import get_rag_manager

    rag = get_rag_manager()
    if rag is None:
        raise HistoryImportError(
            "RAG store (ChromaDB) is unavailable — make sure it's running and reachable"
        )

    docs = []
    for conv in conversations:
        for i, chunk in enumerate(_chunk_conversation(conv)):
            metadata = {
                "type": "imported_conversation",
                "source": conv.source,
                "title": conv.title,
                "external_id": conv.external_id,
                "chunk": i,
                "owner": owner or "",
            }
            docs.append((chunk, metadata))

    added = 0
    if docs:
        result = rag.add_documents_batch(docs)
        if isinstance(result, dict) and result.get("success") is False:
            raise HistoryImportError(result.get("message") or "RAG ingestion failed")
        added = result.get("added_count", len(docs)) if isinstance(result, dict) else len(docs)

    return {
        "conversations": len(conversations),
        "chunks": len(docs),
        "added": added,
    }
