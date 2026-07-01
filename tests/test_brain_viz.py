"""Brain visualizer — projection math and owner-scoped point gathering."""
import sys
import types

import pytest

sklearn = pytest.importorskip("sklearn")

from src.brain_viz import _label_for, _project  # noqa: E402


def test_project_returns_requested_dims():
    vectors = [[float(i + j) for j in range(16)] for i in range(12)]
    coords2 = _project(vectors, 2)
    assert len(coords2) == len(vectors)
    assert all(len(c) == 2 for c in coords2)

    coords3 = _project(vectors, 3)
    assert all(len(c) == 3 for c in coords3)


def test_project_handles_empty_and_tiny_input():
    assert _project([], 2) == []
    tiny = _project([[1.0, 2.0], [3.0, 4.0]], 2)
    assert len(tiny) == 2
    assert all(len(c) == 2 for c in tiny)


def test_label_for_maps_imported_conversation_and_defaults():
    assert _label_for({"type": "imported_conversation", "source": "chatgpt"}) == "imported:chatgpt"
    assert _label_for({"type": "imported_conversation", "source": "claude"}) == "imported:claude"
    assert _label_for({"type": "personal_document"}) == "personal_document"
    assert _label_for({}) == "memory"
    assert _label_for(None) == "memory"


class _FakeCollection:
    def __init__(self, rows):
        self.rows = rows  # list of (id, embedding, doc, meta)

    def get(self, where=None, include=None, limit=None):
        rows = self.rows
        if where and "owner" in where:
            rows = [r for r in rows if r[3].get("owner") == where["owner"]]
        if limit:
            rows = rows[:limit]
        return {
            "ids": [r[0] for r in rows],
            "embeddings": [r[1] for r in rows],
            "documents": [r[2] for r in rows],
            "metadatas": [r[3] for r in rows],
        }


class _FakeLane:
    def __init__(self, name, rows):
        self.name = name
        self.collection = _FakeCollection(rows)

    def encode(self, texts):
        return [[float((hash(t) % 1000)) for _ in range(8)] for t in texts]


@pytest.fixture
def fake_stores(monkeypatch):
    rows_memory = [
        ("m1", [1.0] * 8, "Remember to buy milk", {"owner": "dani", "type": ""}),
        ("m2", [1.1] * 8, "Remember the meeting at 5pm", {"owner": "dani", "type": ""}),
        ("m3", [9.0] * 8, "Someone else's memory", {"owner": "other", "type": ""}),
    ]
    rows_rag = [
        ("r1", [5.0] * 8, "Chat about python decorators",
         {"owner": "dani", "type": "imported_conversation", "source": "chatgpt", "title": "Decorators chat"}),
        ("r2", [5.2] * 8, "Claude chat about rust",
         {"owner": "dani", "type": "imported_conversation", "source": "claude", "title": "Rust chat"}),
    ]

    fake_memory_vector = types.SimpleNamespace(healthy=True, _lanes=[_FakeLane("fastembed", rows_memory)])
    fake_app = types.ModuleType("app")
    fake_app.memory_vector = fake_memory_vector
    monkeypatch.setitem(sys.modules, "app", fake_app)

    fake_rag = types.SimpleNamespace(healthy=True, _lanes=[_FakeLane("rag", rows_rag)])
    import src.rag_singleton as rag_singleton
    monkeypatch.setattr(rag_singleton, "get_rag_manager", lambda: fake_rag)

    from src.brain_viz import _cache
    _cache.clear()
    yield
    _cache.clear()


def test_get_points_merges_lanes_and_scopes_by_owner(fake_stores):
    from src.brain_viz import get_points

    result = get_points("dani", dims=2, limit=100, force=True)
    assert result["count"] == 4  # m1, m2, r1, r2 — "other"-owned m3 excluded

    ids = {p["id"] for p in result["points"]}
    assert ids == {"m1", "m2", "r1", "r2"}

    categories = {p["id"]: p["category"] for p in result["points"]}
    assert categories["r1"] == "imported:chatgpt"
    assert categories["r2"] == "imported:claude"
    assert categories["m1"] == "memory"


def test_search_returns_ranked_matches(fake_stores):
    from src.brain_viz import search

    result = search("dani", "milk", k=2)
    assert result["query"] == "milk"
    assert result["matches"]
    assert all("id" in m and "similarity" in m for m in result["matches"])


def test_search_empty_query_returns_no_matches(fake_stores):
    from src.brain_viz import search

    result = search("dani", "   ", k=5)
    assert result["matches"] == []
