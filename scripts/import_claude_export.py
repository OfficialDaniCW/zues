#!/usr/bin/env python3
"""Import a Claude.ai data export into Odysseus/Zeus (stdlib-only).

Imports:
  - conversations.json  -> chat sessions + messages (SQLite)
  - memories.json       -> memory.json entries
  - projects/*.json     -> document library entries
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1].parent / "data"
IMPORT_SOURCE = "claude-import"
IMPORT_MODEL = "claude"
IMPORT_ENDPOINT = "import://claude"
DEFAULT_FOLDER = "Claude"
DEFAULT_OWNER = "admin"


def parse_iso_ts(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def normalize_role(sender: str | None) -> str:
    role = (sender or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "ai", "bot", "model"}:
        return "assistant"
    return role or "unknown"


def message_text(message: dict) -> str:
    text = (message.get("text") or "").strip()
    if text:
        return text
    parts: list[str] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            part = str(block.get("text") or "").strip()
            if part:
                parts.append(part)
    return "\n".join(parts).strip()


def infer_language(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".md") or lower.endswith(".markdown"):
        return "markdown"
    if lower.endswith(".py"):
        return "python"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".html") or lower.endswith(".htm"):
        return "html"
    if lower.endswith(".csv"):
        return "csv"
    return "markdown"


def split_memory_sections(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("**") and line.endswith("**") and current:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = [line]
        else:
            current.append(line)
    tail = "\n".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks or ([text.strip()] if text.strip() else [])


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            endpoint_url TEXT NOT NULL,
            model TEXT NOT NULL,
            owner TEXT,
            rag BOOLEAN DEFAULT 0,
            archived BOOLEAN DEFAULT 0,
            folder TEXT,
            headers TEXT DEFAULT '{}',
            created_at DATETIME,
            updated_at DATETIME,
            last_accessed DATETIME,
            last_message_at DATETIME,
            is_important BOOLEAN DEFAULT 0,
            message_count INTEGER DEFAULT 0,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            mode TEXT,
            crew_member_id TEXT
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            timestamp DATETIME,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            title TEXT NOT NULL,
            language TEXT,
            current_content TEXT NOT NULL DEFAULT '',
            version_count INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT 1,
            archived BOOLEAN DEFAULT 0,
            owner TEXT,
            created_at DATETIME,
            updated_at DATETIME
        );
        CREATE TABLE IF NOT EXISTS document_versions (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            source TEXT DEFAULT 'ai',
            created_at DATETIME,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """
    )


def import_conversations(conn: sqlite3.Connection, export_dir: Path, owner: str, folder: str, dry_run: bool) -> dict[str, int]:
    conversations = json.loads((export_dir / "conversations.json").read_text(encoding="utf-8"))
    existing = {row[0] for row in conn.execute("SELECT id FROM sessions")}
    stats = {"sessions_created": 0, "sessions_skipped": 0, "messages_created": 0}

    for conv in conversations:
        session_id = str(conv.get("uuid") or uuid.uuid4())
        if session_id in existing:
            stats["sessions_skipped"] += 1
            continue

        name = (conv.get("name") or "Untitled Claude chat").strip() or "Untitled Claude chat"
        created_at = parse_iso_ts(conv.get("created_at")) or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        updated_at = parse_iso_ts(conv.get("updated_at")) or created_at
        messages = conv.get("chat_messages") or []

        if dry_run:
            stats["sessions_created"] += 1
            stats["messages_created"] += sum(1 for m in messages if message_text(m))
            continue

        conn.execute(
            """
            INSERT INTO sessions (
                id, name, endpoint_url, model, owner, rag, archived, folder, headers,
                created_at, updated_at, last_accessed, last_message_at,
                is_important, message_count
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, '{}', ?, ?, ?, ?, 0, 0)
            """,
            (session_id, name[:500], IMPORT_ENDPOINT, IMPORT_MODEL, owner, folder, created_at, updated_at, updated_at, updated_at),
        )

        msg_count = 0
        last_ts = updated_at
        for message in messages:
            content = message_text(message)
            if not content:
                continue
            role = normalize_role(message.get("sender"))
            if role not in {"user", "assistant", "system"}:
                continue
            ts = parse_iso_ts(message.get("created_at")) or created_at
            metadata = json.dumps(
                {
                    "timestamp": f"{ts.replace(' ', 'T')}Z" if ts else None,
                    "source": IMPORT_SOURCE,
                    "claude_message_uuid": message.get("uuid"),
                }
            )
            conn.execute(
                """
                INSERT INTO chat_messages (id, session_id, role, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(message.get("uuid") or uuid.uuid4()), session_id, role, content, metadata, ts),
            )
            msg_count += 1
            last_ts = ts

        conn.execute(
            "UPDATE sessions SET message_count = ?, last_message_at = ? WHERE id = ?",
            (msg_count, last_ts, session_id),
        )
        stats["sessions_created"] += 1
        stats["messages_created"] += msg_count

    if not dry_run:
        conn.commit()
    return stats


def import_memories(export_dir: Path, owner: str, data_dir: Path, dry_run: bool) -> dict[str, int]:
    path = export_dir / "memories.json"
    if not path.exists():
        return {"memories_added": 0}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return {"memories_added": 0}

    record = payload[0]
    project_names: dict[str, str] = {}
    projects_dir = export_dir / "projects"
    if projects_dir.is_dir():
        for project_file in projects_dir.glob("*.json"):
            try:
                project = json.loads(project_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            project_names[str(project.get("uuid") or project_file.stem)] = str(project.get("name") or project_file.stem)

    candidates: list[tuple[str, str, str]] = []
    for chunk in split_memory_sections(str(record.get("conversations_memory") or "")):
        candidates.append((chunk, "fact", "claude-conversations-memory"))
    project_memories = record.get("project_memories") or {}
    if isinstance(project_memories, dict):
        for project_id, text in project_memories.items():
            project_name = project_names.get(str(project_id), str(project_id))
            for chunk in split_memory_sections(str(text or "")):
                candidates.append((chunk, "project", f"claude-project:{project_name}"))

    if dry_run:
        return {"memories_added": len(candidates)}

    memory_file = data_dir / "memory.json"
    existing = json.loads(memory_file.read_text(encoding="utf-8")) if memory_file.exists() else []
    if not isinstance(existing, list):
        existing = []
    seen = {re.sub(r"\s+", " ", entry.get("text", "").strip().lower()) for entry in existing if isinstance(entry, dict)}
    added = 0
    now = int(datetime.utcnow().timestamp())
    for text, category, source in candidates:
        key = re.sub(r"\s+", " ", text.strip().lower())
        if not key or key in seen:
            continue
        existing.append(
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "timestamp": now,
                "source": source,
                "category": category,
                "uses": 0,
                "owner": owner,
            }
        )
        seen.add(key)
        added += 1
    if added:
        memory_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"memories_added": added}


def import_project_docs(conn: sqlite3.Connection, export_dir: Path, owner: str, dry_run: bool) -> dict[str, int]:
    projects_dir = export_dir / "projects"
    if not projects_dir.is_dir():
        return {"documents_created": 0, "documents_skipped": 0}

    existing_titles = {row[0] for row in conn.execute("SELECT title FROM documents WHERE owner = ?", (owner,))}
    stats = {"documents_created": 0, "documents_skipped": 0}
    for project_file in sorted(projects_dir.glob("*.json")):
        try:
            project = json.loads(project_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        project_name = str(project.get("name") or project_file.stem).strip()
        for doc in project.get("docs") or []:
            filename = str(doc.get("filename") or "Untitled").strip()
            content = str(doc.get("content") or "")
            if not content.strip():
                continue
            title = f"Claude · {project_name} · {filename}"
            if title in existing_titles:
                stats["documents_skipped"] += 1
                continue
            if dry_run:
                stats["documents_created"] += 1
                continue
            doc_id = str(doc.get("uuid") or uuid.uuid4())
            created_at = parse_iso_ts(doc.get("created_at")) or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
            conn.execute(
                """
                INSERT INTO documents (
                    id, session_id, title, language, current_content, version_count,
                    is_active, archived, owner, created_at, updated_at
                ) VALUES (?, NULL, ?, ?, ?, 1, 1, 0, ?, ?, ?)
                """,
                (doc_id, title[:500], infer_language(filename), content, owner, created_at, created_at),
            )
            conn.execute(
                """
                INSERT INTO document_versions (
                    id, document_id, version_number, content, summary, source, created_at
                ) VALUES (?, ?, 1, ?, ?, 'user', ?)
                """,
                (str(uuid.uuid4()), doc_id, content, "Imported from Claude project export", created_at),
            )
            existing_titles.add(title)
            stats["documents_created"] += 1
    if not dry_run:
        conn.commit()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Claude.ai export data into Odysseus/Zeus")
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    export_dir = args.export_dir.resolve()
    data_dir = args.data_dir.resolve()
    if not export_dir.is_dir():
        raise SystemExit(f"Export directory not found: {export_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    db_path = data_dir / "app.db"
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        print(f"Import source: {export_dir}")
        print(f"Data directory: {data_dir}")
        if args.dry_run:
            print("Dry run only — no writes.")
        conv_stats = import_conversations(conn, export_dir, args.owner, args.folder, args.dry_run)
        mem_stats = import_memories(export_dir, args.owner, data_dir, args.dry_run)
        doc_stats = import_project_docs(conn, export_dir, args.owner, args.dry_run)
    finally:
        conn.close()

    print(
        f"Conversations: {conv_stats['sessions_created']} sessions, "
        f"{conv_stats['messages_created']} messages ({conv_stats['sessions_skipped']} skipped)"
    )
    print(f"Memories: {mem_stats['memories_added']} added")
    print(f"Documents: {doc_stats['documents_created']} created ({doc_stats.get('documents_skipped', 0)} skipped)")
    if not args.dry_run:
        print("Refresh Odysseus at http://localhost:7000 to see imported chats under the Claude folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
