from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RankedDocument:
    """A single document returned by the retriever agent.

    Attributes:
        chunk_id: The id of the retrieved chunk (the only id the harness carries).
        doc_id: Alias of chunk_id; the harness has no separate document id.
        text: The snippet the retriever saw during search.
        score: Backend relevance score, passed through from the search client.
        rank: 1-based position in the final ranking.
        retrieval_query: The query that first surfaced this chunk.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int = 0
    retrieval_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document.

        Returns:
            The document as a plain dict.
        """
        return asdict(self)


@dataclass
class RetrievalResult:
    """The complete result of one ``retrieve()`` call.

    ``documents`` is the ranking; the remaining fields are per-run telemetry and the
    raw transcript for inspection. ``termination_reason`` is one of ``finalized``,
    ``max_rounds_reached`` (round ceiling exhausted, or a round hit the turn cap),
    ``llm_error``, ``free_text_loop``, ``search_lock_loop``.
    """

    query: str
    documents: list[RankedDocument]
    total_search_calls: int = 0
    total_search_turns: int = 0
    total_finalize_calls: int = 0
    rounds_completed: int = 0
    termination_reason: str = ""
    save_and_advance_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    all_round_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    persistent_saved_final: list[dict[str, Any]] = field(default_factory=list)
    round_summaries: list[dict[str, Any]] = field(default_factory=list)
    searches_per_round: list[int] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    retrieved_docids: list[str] = field(default_factory=list)
    final_coverage_declaration: dict[str, Any] = field(default_factory=dict)
    finalize_reject_count: int = 0
    search_duplicate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result, including the full transcript.

        Returns:
            The result as a plain dict.
        """
        return asdict(self)


@dataclass
class ContextBudget:
    """Tracks token usage against a per-round budget.

    Attributes:
        max_tokens: The per-round token budget.
        hard_lock_ratio: Fraction of the budget at which search locks.
        current_tokens: The current token count for the round.
    """

    max_tokens: int
    hard_lock_ratio: float
    current_tokens: int = 0

    @property
    def hard_locked(self) -> bool:
        """Whether the current token count is at or past the lock threshold."""
        return self.current_tokens >= int(self.max_tokens * self.hard_lock_ratio)
