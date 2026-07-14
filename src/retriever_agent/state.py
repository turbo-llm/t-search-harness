from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SavedChunk:
    """A chunk the model saved via save_and_advance, carried across rounds."""

    chunk_id: str
    snippet: str
    reason: str
    round_saved: int
    score: float = 0.0


@dataclass
class SessionState:
    """Mutable state for one ``retrieve()`` call.

    Cross-round fields persist for the whole session; the per-round fields are reset
    on each accepted save_and_advance.
    """

    persistent_saved: dict[str, SavedChunk] = field(default_factory=dict)
    global_seen_chunk_ids: set[str] = field(default_factory=set)
    round_summaries: list[dict[str, Any]] = field(default_factory=list)
    current_round: int = 1
    chunk_score_cache: dict[str, float] = field(default_factory=dict)
    chunk_snippet_cache: dict[str, str] = field(default_factory=dict)
    chunk_first_query: dict[str, str] = field(default_factory=dict)
    seen_this_round: set[str] = field(default_factory=set)
    searches_this_round: int = 0
    consecutive_zero_new_this_round: int = 0
    searches_per_round: list[int] = field(default_factory=list)
    locked_search_turns_streak: int = 0
    finalize_reject_count: int = 0
    search_history: set[str] = field(default_factory=set)
    search_history_round: dict[str, int] = field(default_factory=dict)
    search_duplicate_count: int = 0
    final_coverage: dict[str, Any] = field(default_factory=dict)
