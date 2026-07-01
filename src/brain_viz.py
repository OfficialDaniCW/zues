"""
brain_viz.py

Server-side embedding-space projection for the "brain visualizer" (/brain) —
pulls the owner's already-stored vectors out of Chroma (memory + RAG, which
now also holds imported ChatGPT/Claude conversations and skills) and reduces
them to 2D/3D for an interactive scatter in the frontend.

The PCA -> t-SNE shape follows mtybadger/chromaviz (MIT): pull embeddings,
denoise with PCA, then t-SNE to the render dimension. Reimplemented natively
here rather than vendored, since chromaviz itself is a small Flask+React+
Three.js app (unmaintained since 2023) with no search feature — this app is
FastAPI + vanilla JS with no React/Three.js in the stack, and adds live
query search: embedding the query and ranking cosine similarity against the
*original* high-dimensional vectors (2D distances don't preserve semantic
distance well enough to search on directly), returned as highlight ids
against the already-rendered projection instead of recomputing it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_POINTS = 3000
CACHE_TTL_SECONDS = 300
MIN_POINTS_FOR_TSNE = 4

# owner -> (expires_at, {"points": [...], "ids": [...], "vectors": [...], "lanes": {...}})
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class BrainVizError(RuntimeError):
    pass


def _require_sklearn():
    try:
        import numpy  # noqa: F401
        from sklearn.decomposition import PCA  # noqa: F401
        from sklearn.manifold import TSNE  # noqa: F401
    except ImportError as e:
        raise BrainVizError(
            "Brain visualizer needs scikit-learn. Install it with: "
            "pip install scikit-learn (see requirements-optional.txt)"
        ) from e


def _label_for(metadata: Dict[str, Any]) -> str:
    """Fixed, small category set so the frontend can assign categorical
    colors in a stable order (see the dataviz palette: identity color is
    assigned by fixed slot, never generated per-value)."""
    mtype = (metadata or {}).get("type") or ""
    source = (metadata or {}).get("source") or ""
    if mtype == "imported_conversation":
        return f"imported:{source}" if source else "imported"
    if mtype:
        return mtype
    return "memory"


def _gather_points(owner: Optional[str], limit: int) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
    """Pull (ids, vectors, records) from every embedding lane this app knows
    about — memory_vector's collection and the RAG store's collection(s).
    Reaches into each store's `._lanes` (a list of EmbeddingLane) directly;
    both stores already expose the raw Chroma collection there and there's
    no public "dump everything" API to add for a single read-only view."""
    from app import memory_vector  # deferred: avoids a circular import at module load

    ids: List[str] = []
    vectors: List[List[float]] = []
    records: List[Dict[str, Any]] = []
    seen_ids = set()

    lanes = []
    if memory_vector is not None and getattr(memory_vector, "healthy", False):
        lanes.extend(memory_vector._lanes)

    try:
        from src.rag_singleton import get_rag_manager
        rag = get_rag_manager()
        if rag is not None and getattr(rag, "healthy", False):
            lanes.extend(rag._lanes)
    except Exception as e:
        logger.debug("brain_viz: RAG lane unavailable: %s", e)

    where = {"owner": owner} if owner else None
    for lane in lanes:
        if len(ids) >= limit:
            break
        try:
            got = lane.collection.get(
                where=where,
                include=["embeddings", "documents", "metadatas"],
                limit=max(0, limit - len(ids)),
            )
        except Exception as e:
            logger.warning("brain_viz: lane %s get() failed: %s", getattr(lane, "name", "?"), e)
            continue

        lane_ids = got.get("ids") or []
        lane_embeds = got.get("embeddings")
        lane_docs = got.get("documents") or []
        lane_metas = got.get("metadatas") or []
        if lane_embeds is None:
            continue
        for i, doc_id in enumerate(lane_ids):
            if doc_id in seen_ids:
                continue
            vec = lane_embeds[i]
            if vec is None or len(vec) == 0:
                continue
            seen_ids.add(doc_id)
            meta = lane_metas[i] if i < len(lane_metas) else {}
            text = (lane_docs[i] if i < len(lane_docs) else "") or ""
            ids.append(doc_id)
            vectors.append(list(vec))
            records.append({
                "id": doc_id,
                "text": text,
                "preview": text[:220],
                "category": _label_for(meta or {}),
                "title": (meta or {}).get("title") or "",
                "source": (meta or {}).get("source") or "",
            })

    return ids, vectors, records


def _project(vectors: List[List[float]], dims: int) -> List[List[float]]:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(vectors, dtype=float)
    n = X.shape[0]
    if n < MIN_POINTS_FOR_TSNE:
        # Not enough points for a meaningful t-SNE layout — spread them out
        # deterministically instead of erroring on a fresh/near-empty brain.
        return [[float(i), 0.0, 0.0][:dims] for i in range(n)]

    pca_dims = min(50, n - 1, X.shape[1])
    if pca_dims >= dims:
        X = PCA(n_components=pca_dims, random_state=42).fit_transform(X)

    perplexity = max(2, min(30, n - 1))
    tsne = TSNE(n_components=dims, perplexity=perplexity, random_state=42, init="pca")
    return tsne.fit_transform(X).tolist()


def get_points(owner: Optional[str], dims: int = 2, limit: int = MAX_POINTS, force: bool = False) -> Dict[str, Any]:
    """Owner-scoped projection, cached for CACHE_TTL_SECONDS so a search
    query doesn't force a full PCA/t-SNE recompute on every keystroke."""
    _require_sklearn()
    dims = 3 if dims == 3 else 2
    limit = max(1, min(int(limit or MAX_POINTS), MAX_POINTS))
    cache_key = f"{owner or ''}:{dims}:{limit}"

    now = time.time()
    if not force:
        cached = _cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

    ids, vectors, records = _gather_points(owner, limit)
    coords = _project(vectors, dims)

    points = []
    for rec, coord in zip(records, coords):
        p = dict(rec)
        p["x"] = coord[0]
        p["y"] = coord[1] if len(coord) > 1 else 0.0
        if dims == 3:
            p["z"] = coord[2] if len(coord) > 2 else 0.0
        points.append(p)

    result = {
        "points": points,
        "count": len(points),
        "dims": dims,
        "computed_at": now,
    }
    _cache[cache_key] = (now + CACHE_TTL_SECONDS, {**result, "_ids": ids, "_vectors": vectors})
    return result


def search(owner: Optional[str], query: str, k: int = 15, dims: int = 2) -> Dict[str, Any]:
    """Rank the cached point set by cosine similarity to `query`, embedded
    with the same lane encoder used to store them. Returns ids + similarity
    so the frontend highlights matches on the already-rendered scatter
    instead of re-projecting."""
    import numpy as np

    query = (query or "").strip()
    if not query:
        return {"query": query, "matches": []}

    dims = 3 if dims == 3 else 2
    limit = MAX_POINTS
    cache_key = f"{owner or ''}:{dims}:{limit}"
    cached = _cache.get(cache_key)
    if not cached or cached[0] <= time.time():
        get_points(owner, dims=dims, limit=limit)  # populate cache
        cached = _cache.get(cache_key)
    if not cached:
        return {"query": query, "matches": []}

    payload = cached[1]
    ids = payload.get("_ids") or []
    vectors = payload.get("_vectors") or []
    if not ids or not vectors:
        return {"query": query, "matches": []}

    from app import memory_vector
    encoder = memory_vector._lanes[0] if memory_vector and memory_vector._lanes else None
    if encoder is None:
        return {"query": query, "matches": []}
    q_vec = np.asarray(encoder.encode([query])[0], dtype=float)

    V = np.asarray(vectors, dtype=float)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    v_norms = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    sims = v_norms @ q_norm

    k = max(1, min(int(k or 15), len(ids)))
    top_idx = np.argsort(-sims)[:k]
    matches = [
        {"id": ids[i], "similarity": round(float(sims[i]), 4)}
        for i in top_idx
    ]
    return {"query": query, "matches": matches}
