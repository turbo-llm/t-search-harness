from __future__ import annotations

from typing import Any

from .state import SessionState
from .types import RankedDocument


def build_final_ranking(
    finalized_ranking: list[dict[str, Any]], state: SessionState, max_results: int
) -> list[RankedDocument]:
    """Resolve the final ranked document list.

    Priority:
        1. The model's accepted ``finalize_ranking`` list, if any.
        2. Fallback: the persistent saved set in insertion order, then any globally
           seen chunk by score, up to ``max_results``.

    Args:
        finalized_ranking: The normalized ranking from an accepted finalize (may be empty).
        state: The session state carrying saved chunks, seen chunks, and caches.
        max_results: The maximum number of documents to return.

    Returns:
        The ranked documents; each carries the snippet the retriever saw.
    """
    ordered_ids: list[str] = []

    if finalized_ranking:
        for item in finalized_ranking:
            cid = item["chunk_id"]
            if cid not in ordered_ids:
                ordered_ids.append(cid)
    else:
        for cid in state.persistent_saved:
            if cid not in ordered_ids:
                ordered_ids.append(cid)
        if len(ordered_ids) < max_results:
            extras = sorted(
                (cid for cid in state.global_seen_chunk_ids if cid not in ordered_ids),
                key=lambda c: state.chunk_score_cache.get(c, 0.0),
                reverse=True,
            )
            for cid in extras:
                if len(ordered_ids) >= max_results:
                    break
                ordered_ids.append(cid)

    docs: list[RankedDocument] = []
    for rank, cid in enumerate(ordered_ids[:max_results], 1):
        snippet = (
            state.persistent_saved[cid].snippet
            if cid in state.persistent_saved
            else state.chunk_snippet_cache.get(cid, "")
        )
        docs.append(
            RankedDocument(
                chunk_id=cid,
                doc_id=cid,
                text=snippet,
                score=state.chunk_score_cache.get(cid, 0.0),
                rank=rank,
                retrieval_query=state.chunk_first_query.get(cid, ""),
            )
        )
    return docs
