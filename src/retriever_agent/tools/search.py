from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from ..clients.search import SearchClient
from ..config import AgentConfig
from ..state import SessionState
from .base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)

_SEARCH_BACKEND_RETRIES = 3


class SearchTool(Tool):
    """search_corpus: semantic search with saved-set/within-round dedup and hard-lock/duplicate-query gates."""

    name = "search_corpus"

    def __init__(self, config: AgentConfig, tool_schema: dict[str, Any]) -> None:
        super().__init__(config, tool_schema)
        props = tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
        default_top_k = props.get("top_k", {}).get("default", 5)
        self._default_top_k = default_top_k if isinstance(default_top_k, int) else 5

    def _coerce_top_k(self, raw: Any) -> int:
        """Coerce a model-supplied top_k value.

        Args:
            raw: The value from the tool-call arguments.

        Returns:
            The value if it is a positive int, else the schema default.
        """
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            return raw
        return self._default_top_k

    def gate(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult | None:
        """Reject the search when the budget is locked or the query repeats this session.

        A query that passes the duplicate check is recorded into the session
        search history as a side effect.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            A ToolResult to reject the call, or None to proceed to execute().
        """
        if ctx.hard_locked:
            lock_pct = int(self._config.hard_lock_ratio * 100)
            lock_msg = (
                f"Round budget at {lock_pct}%+. search_corpus is locked for this round. "
                "Required: call save_and_advance (saved_chunks, drop_from_saved, "
                "covered_concepts, unresolved_concepts, round_summary, next_round_goal) "
                "to advance to the next round with fresh context, OR call finalize_ranking "
                "(ranking, covered_concepts, unresolved_concepts) if current evidence is sufficient."
            )
            payload = json.dumps(
                {
                    "error": lock_msg,
                    "is_last_round": ctx.is_last_round,
                    "saved_set_size": len(state.persistent_saved),
                },
                ensure_ascii=False,
            )
            return ToolResult(payload=payload, locked_search=True)

        q_raw = str(args.get("query", "")).strip()
        norm = " ".join(q_raw.lower().split())
        if norm and norm in state.search_history:
            state.search_duplicate_count += 1
            state.consecutive_zero_new_this_round += 1
            prev_round = state.search_history_round.get(norm, 0)
            payload = json.dumps(
                {
                    "error": "duplicate_query",
                    "note": (
                        f"Query '{q_raw}' was already issued in round {prev_round}. "
                        "Vary terms, paraphrase, or pivot to a different angle."
                    ),
                },
                ensure_ascii=False,
            )
            return ToolResult(payload=payload)
        if norm:
            state.search_history.add(norm)
            state.search_history_round[norm] = state.current_round
        return None

    def execute(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult:
        """Run the search and record accepted hits into the session state.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            The ToolResult carrying the model-visible search results.
        """
        return ToolResult(payload=self._search(args, state, ctx.search), accepted=True)

    def _search(self, args: dict[str, Any], state: SessionState, search: SearchClient) -> str:
        """Call the backend and format accepted hits for the model.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            search: The injected search client.

        Returns:
            The JSON string the model sees as the tool result.
        """
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "Empty query."}, ensure_ascii=False)
        top_k = self._coerce_top_k(args.get("top_k"))

        state.searches_this_round += 1

        last_err: str | None = None
        raw_json = None
        for attempt in range(_SEARCH_BACKEND_RETRIES):
            try:
                raw_json = search.search(query, top_k)
                break
            except Exception as exc:  # noqa: BLE001 — surface any backend error to the model
                last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
                wait = min(2**attempt + random.random(), 10)
                log.warning(
                    "search backend call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    _SEARCH_BACKEND_RETRIES,
                    last_err,
                    wait,
                )
                time.sleep(wait)
        if raw_json is None:
            return json.dumps(
                {
                    "error": (
                        "Search backend unavailable after retries. Try a different query, "
                        "call save_and_advance with what you have, or finalize_ranking."
                    ),
                    "backend_error": last_err,
                },
                ensure_ascii=False,
            )
        try:
            results = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"error": "Backend search returned invalid JSON."}, ensure_ascii=False)
        if not isinstance(results, list):
            return json.dumps({"error": "Backend search returned invalid JSON."}, ensure_ascii=False)

        out: list[dict[str, Any]] = []
        new_this_turn = 0
        filtered_already_saved = 0
        filtered_seen_this_round = 0

        for r in results:
            if not isinstance(r, dict):
                continue
            docid = str(r.get("docid", ""))
            if not docid:
                continue
            snippet = r.get("snippet", "")
            score = r.get("score", 0.0)

            if docid in state.persistent_saved:
                filtered_already_saved += 1
                continue
            if docid in state.seen_this_round:
                filtered_seen_this_round += 1
                continue

            state.seen_this_round.add(docid)
            state.global_seen_chunk_ids.add(docid)
            try:
                state.chunk_score_cache[docid] = float(score)
            except (TypeError, ValueError):
                state.chunk_score_cache[docid] = 0.0
            if docid not in state.chunk_snippet_cache:
                state.chunk_snippet_cache[docid] = snippet
                state.chunk_first_query[docid] = query
            out.append({"chunk_id": docid, "snippet": snippet, "score": score})
            new_this_turn += 1

        if new_this_turn == 0:
            state.consecutive_zero_new_this_round += 1
        else:
            state.consecutive_zero_new_this_round = 0

        payload: dict[str, Any] = {
            "results": out,
            "count": len(out),
            "new_this_turn": new_this_turn,
        }
        if filtered_already_saved or filtered_seen_this_round:
            payload["filtered"] = {
                "already_in_saved": filtered_already_saved,
                "already_seen_this_round": filtered_seen_this_round,
            }
        return json.dumps(payload, ensure_ascii=False)
