"""
telegram_bridge.py

A Telegram bot as a second front door into one Zeus account, with the same
tool access as the web UI (full agent loop — shell, files, email, etc).

Pairing model: the admin sets the bot token, then starts a short pairing
window. The next message the bot receives during that window binds its
Telegram chat_id as the *only* chat this bridge will ever respond to.
Every other sender — anyone who finds the bot's username on Telegram — is
silently ignored. There is no per-message auth beyond that chat_id match,
so the token and the pairing are the entire trust boundary; see
THREAT_MODEL.md for what "admin" access means once paired.

The bot token lives encrypted at rest (src/secret_storage.py, the same
mechanism used for IMAP/SMTP passwords) in data/telegram.json — never in
.env or anywhere git-tracked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Optional

import httpx

from core.atomic_io import atomic_write_json
from src.constants import AUTH_FILE, DATA_DIR
from src.secret_storage import decrypt as _dec
from src.secret_storage import encrypt as _enc

logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(DATA_DIR, "telegram.json")
_API_BASE = "https://api.telegram.org/bot{token}"
_TELEGRAM_MAX_LEN = 4000  # stay under Telegram's 4096 hard cap
_DEFAULT_PAIRING_MINUTES = 5
_POLL_TIMEOUT = 25

SYSTEM_PROMPT = (
    "You are Zeus, the user's personal AI agent, reached here over their "
    "private Telegram bridge. You have the same tool access as the web UI "
    "(shell, files, email, calendar, memory, etc). This is a phone chat "
    "app, not a document editor — keep replies concise."
)

_poller_task: Optional[asyncio.Task] = None


# --------------------------------------------------------------------------
# Config persistence
# --------------------------------------------------------------------------

def _default_config() -> dict:
    return {
        "token_enc": "",
        "enabled": False,
        "owner": None,
        "chat_id": None,
        "session_id": None,
        "pairing_until": 0,
        "update_offset": 0,
    }


def _load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return _default_config()
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        cfg = _default_config()
        if isinstance(data, dict):
            cfg.update(data)
        return cfg
    except Exception as e:
        logger.warning("telegram config load failed: %s", e)
        return _default_config()


def _save_config(cfg: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    atomic_write_json(CONFIG_FILE, cfg)


def _token(cfg: dict) -> str:
    return _dec(cfg.get("token_enc") or "")


def _resolve_default_owner() -> Optional[str]:
    """Mirror app.py's primary-owner resolution: the first admin account in
    auth.json, else the first account."""
    try:
        with open(AUTH_FILE, encoding="utf-8") as f:
            users = json.load(f).get("users", {})
    except Exception:
        return None
    for uname, udata in users.items():
        if isinstance(udata, dict) and udata.get("is_admin") is True:
            return uname
    return next(iter(users), None)


# --------------------------------------------------------------------------
# Telegram Bot API
# --------------------------------------------------------------------------

async def _api_call(token: str, method: str, http_timeout: float = 35.0, **params) -> dict:
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        r = await client.post(f"{_API_BASE.format(token=token)}/{method}", json=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("description") or f"Telegram API error ({method})")
        return data["result"]


async def verify_token(token: str) -> dict:
    """Raises if the token is invalid; returns the bot's getMe payload."""
    return await _api_call(token.strip(), "getMe")


async def _send(token: str, chat_id: int, text: str) -> None:
    text = text or "(empty response)"
    for i in range(0, len(text), _TELEGRAM_MAX_LEN):
        chunk = text[i:i + _TELEGRAM_MAX_LEN]
        try:
            await _api_call(token, "sendMessage", chat_id=chat_id, text=chunk)
        except Exception as e:
            logger.error("telegram sendMessage failed: %s", e)
            break


# --------------------------------------------------------------------------
# Admin-facing operations (called from routes/telegram_routes.py)
# --------------------------------------------------------------------------

def get_status() -> dict:
    cfg = _load_config()
    return {
        "configured": bool(cfg.get("token_enc")),
        "enabled": bool(cfg.get("enabled")),
        "owner": cfg.get("owner"),
        "paired": cfg.get("chat_id") is not None,
        "pairing_active": time.time() < float(cfg.get("pairing_until") or 0),
    }


async def set_token(token: str) -> dict:
    token = (token or "").strip()
    if not token:
        raise ValueError("token is required")
    bot = await verify_token(token)
    cfg = _load_config()
    cfg["token_enc"] = _enc(token)
    cfg["owner"] = cfg.get("owner") or _resolve_default_owner()
    cfg["enabled"] = True
    _save_config(cfg)
    ensure_poller_running()
    return {"username": bot.get("username"), "id": bot.get("id")}


def start_pairing(minutes: int = _DEFAULT_PAIRING_MINUTES) -> dict:
    cfg = _load_config()
    if not cfg.get("token_enc"):
        raise ValueError("Set the bot token first")
    cfg["chat_id"] = None
    cfg["session_id"] = None
    cfg["pairing_until"] = time.time() + max(30, minutes * 60)
    cfg["owner"] = cfg.get("owner") or _resolve_default_owner()
    cfg["enabled"] = True
    _save_config(cfg)
    ensure_poller_running()
    return {"pairing_until": cfg["pairing_until"]}


def enable() -> None:
    cfg = _load_config()
    if not cfg.get("token_enc"):
        raise ValueError("Set the bot token first")
    cfg["enabled"] = True
    _save_config(cfg)
    ensure_poller_running()


def disable() -> None:
    cfg = _load_config()
    cfg["enabled"] = False
    _save_config(cfg)


async def send_test_message() -> None:
    cfg = _load_config()
    token = _token(cfg)
    if not token or not cfg.get("chat_id"):
        raise ValueError("Bot isn't paired yet")
    await _send(token, cfg["chat_id"], "Zeus here — Telegram bridge is working.")


# --------------------------------------------------------------------------
# Agent turn — full tool access, same post-response memory growth as web chat
# --------------------------------------------------------------------------

async def _handle_agent_turn(cfg: dict, text: str) -> str:
    from core.models import ChatMessage
    from routes.chat_helpers import load_prefs_for_user, run_post_response_tasks
    from src.agent_loop import stream_agent_loop
    from src.task_endpoint import resolve_task_candidates

    from app import memory_manager, memory_vector, session_manager, webhook_manager

    owner = cfg["owner"]
    candidates = resolve_task_candidates(owner=owner)
    if not candidates:
        return "No model endpoint is configured yet — set one up in Zeus Settings first."
    endpoint_url, model, headers = candidates[0]
    fallbacks = candidates[1:]

    sess = None
    if cfg.get("session_id"):
        try:
            sess = session_manager.get_session(cfg["session_id"])
        except KeyError:
            sess = None
    if sess is None:
        sid = str(uuid.uuid4())
        sess = session_manager.create_session(
            session_id=sid, name="Telegram", endpoint_url=endpoint_url,
            model=model, owner=owner,
        )
        cfg["session_id"] = sid
        _save_config(cfg)

    sess.add_message(ChatMessage("user", text))
    # Plain role/content dicts only — some OpenAI-compatible APIs reject
    # unrecognized keys, and ChatMessage.to_dict() includes "metadata" when set.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": m.role, "content": m.content} for m in sess.history[-20:]
    )

    full_text = ""
    async for event_str in stream_agent_loop(
        endpoint_url=sess.endpoint_url or endpoint_url,
        model=sess.model or model,
        messages=messages,
        session_id=sess.id,
        owner=owner,
        headers=sess.headers or headers,
        fallbacks=fallbacks,
    ):
        if event_str.startswith("data: ") and not event_str.startswith("data: [DONE]"):
            try:
                data = json.loads(event_str[6:])
            except json.JSONDecodeError:
                continue
            if "delta" in data and not data.get("thinking"):
                full_text += data["delta"]

    full_text = full_text.strip() or "(no response)"
    sess.add_message(ChatMessage("assistant", full_text))
    session_manager.save_sessions()

    try:
        uprefs = load_prefs_for_user(owner)
        run_post_response_tasks(
            sess, session_manager, sess.id, text, full_text, None,
            uprefs, memory_manager, memory_vector, webhook_manager,
            owner=owner,
        )
    except Exception as e:
        logger.warning("telegram post-response tasks failed: %s", e)

    return full_text


# --------------------------------------------------------------------------
# Poll loop
# --------------------------------------------------------------------------

async def _process_update(cfg: dict, update: dict) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat_id = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    if chat_id is None or not text:
        return

    token = _token(cfg)
    if not token:
        return
    pairing_active = time.time() < float(cfg.get("pairing_until") or 0)

    if cfg.get("chat_id") is None:
        if not pairing_active:
            # Not paired and no pairing window open — stay silent. Anyone can
            # find the bot's @username on Telegram; only an admin-initiated
            # pairing window lets a first message bind.
            logger.info("Ignored Telegram message from unpaired chat (no pairing window open)")
            return
        cfg["chat_id"] = chat_id
        cfg["pairing_until"] = 0
        _save_config(cfg)
        await _send(token, chat_id, f"Paired. You're chatting with Zeus as {cfg.get('owner')}.")
        return

    if chat_id != cfg["chat_id"]:
        logger.warning("Ignored Telegram message from unpaired chat_id=%s", chat_id)
        return

    if text == "/start":
        await _send(token, chat_id, "Already paired — send me anything.")
        return

    try:
        reply = await _handle_agent_turn(cfg, text)
    except Exception as e:
        logger.error("telegram agent turn failed: %s", e, exc_info=True)
        reply = f"Error: {e}"
    await _send(token, chat_id, reply)


async def _poll_forever() -> None:
    logger.info("Telegram bridge poller starting")
    while True:
        cfg = _load_config()
        if not cfg.get("enabled") or not cfg.get("token_enc"):
            await asyncio.sleep(15)
            continue
        token = _token(cfg)
        try:
            updates = await _api_call(
                token, "getUpdates",
                http_timeout=_POLL_TIMEOUT + 10,
                offset=cfg.get("update_offset") or 0,
                timeout=_POLL_TIMEOUT,
            )
        except Exception as e:
            logger.warning("telegram getUpdates failed: %s", e)
            await asyncio.sleep(10)
            continue

        for update in updates:
            cfg = _load_config()
            if not cfg.get("enabled"):
                break
            cfg["update_offset"] = int(update["update_id"]) + 1
            _save_config(cfg)
            try:
                await _process_update(cfg, update)
            except Exception as e:
                logger.error("telegram update handling failed: %s", e, exc_info=True)


def ensure_poller_running() -> None:
    global _poller_task
    if _poller_task is not None and not _poller_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop yet — start_poller() at app startup will call again
    _poller_task = loop.create_task(_poll_forever())


def start_poller() -> None:
    """Called once at app startup (or route module setup). No-op if disabled
    or unconfigured — the config/pairing routes call ensure_poller_running()
    again once the admin sets a token."""
    cfg = _load_config()
    if cfg.get("enabled") and cfg.get("token_enc"):
        ensure_poller_running()
