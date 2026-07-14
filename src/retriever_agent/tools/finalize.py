from __future__ import annotations

import json
from typing import Any

from ..state import SessionState
from .base import Tool, ToolContext, ToolResult


class FinalizeTool(Tool):
    """finalize_ranking: submit the final ranked list and end the session."""

    name = "finalize_ranking"

    def execute(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult:
        """Validate the ranking + coverage and, on success, end the session.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            The ToolResult; ``finalized_ranking`` carries the normalized list when accepted.
        """
        accepted, payload, ranking = self._finalize(args, state, is_last_round=ctx.is_last_round)
        if accepted:
            return ToolResult(payload=payload, accepted=True, finalized_ranking=ranking)
        return ToolResult(payload=payload)

    def _finalize(
        self, args: dict[str, Any], state: SessionState, *, is_last_round: bool
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Validate the finalize arguments, normalize the ranking, and record the coverage declaration on accept.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            is_last_round: Whether this is the final round (relaxes the coverage gate).

        Returns:
            ``(accepted, payload_json, normalized_ranking)``; the ranking is empty on reject.
        """
        cfg = self._config
        ranking_arg = args.get("ranking")
        if not isinstance(ranking_arg, list):
            ranking_arg = []

        if len(ranking_arg) == 0:
            empty_payload = {
                "accepted": False,
                "error": (
                    "Empty ranking rejected. Submit your best candidates "
                    "from your current context (saved chunks + this round's seen)."
                ),
                "saved_set_size": len(state.persistent_saved),
                "saved_sample": list(state.persistent_saved.keys())[:10],
            }
            empty_payload["seen_this_round_size"] = len(state.seen_this_round)
            return False, json.dumps(empty_payload, ensure_ascii=False), []

        covered_raw = args.get("covered_concepts")
        unresolved_raw = args.get("unresolved_concepts")
        if not isinstance(covered_raw, list) or not isinstance(unresolved_raw, list):
            return (
                False,
                json.dumps(
                    {
                        "accepted": False,
                        "error": (
                            "covered_concepts and unresolved_concepts are required "
                            "(arrays of strings). Declare which atoms of your query "
                            "decomposition are supported by the final ranking and which "
                            "are still open."
                        ),
                    },
                    ensure_ascii=False,
                ),
                [],
            )

        covered_concepts = [str(x).strip() for x in covered_raw if str(x).strip()]
        unresolved_concepts = [str(x).strip() for x in unresolved_raw if str(x).strip()]

        if not covered_concepts and not unresolved_concepts:
            return (
                False,
                json.dumps(
                    {
                        "accepted": False,
                        "error": (
                            "At least one of covered_concepts or unresolved_concepts "
                            "must be non-empty. Decompose your query and declare coverage."
                        ),
                    },
                    ensure_ascii=False,
                ),
                [],
            )

        concept_overlap = set(covered_concepts) & set(unresolved_concepts)
        if concept_overlap:
            return (
                False,
                json.dumps(
                    {
                        "accepted": False,
                        "error": (
                            "covered_concepts and unresolved_concepts overlap: "
                            f"{sorted(concept_overlap)}. Pick one bucket per concept."
                        ),
                    },
                    ensure_ascii=False,
                ),
                [],
            )

        total_concepts = len(covered_concepts) + len(unresolved_concepts)
        unresolved_ratio = len(unresolved_concepts) / total_concepts if total_concepts else 0.0
        if not is_last_round and unresolved_ratio >= cfg.finalize_unresolved_reject_ratio:
            return (
                False,
                json.dumps(
                    {
                        "accepted": False,
                        "error": (
                            "finalize_ranking rejected: "
                            f"{len(unresolved_concepts)}/{total_concepts} "
                            f"concepts still unresolved ({int(unresolved_ratio * 100)}%). "
                            "Finalize is for when the query is actually covered. "
                            "Call save_and_advance instead to continue in a fresh "
                            "round targeting the unresolved concepts."
                        ),
                        "covered_declared": covered_concepts,
                        "unresolved_declared": unresolved_concepts,
                        "unresolved_ratio": round(unresolved_ratio, 2),
                        "is_last_round": is_last_round,
                    },
                    ensure_ascii=False,
                ),
                [],
            )

        valid_ids = state.seen_this_round | set(state.persistent_saved.keys())
        normalized: list[dict[str, Any]] = []
        invalid_entries: list[str] = []
        short_reasons: list[str] = []
        for item in ranking_arg:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not cid:
                continue
            if cid not in valid_ids:
                invalid_entries.append(cid)
                continue
            if len(reason) < cfg.min_reason_len:
                short_reasons.append(cid)
                continue
            normalized.append({"chunk_id": cid, "reason": reason})

        if not normalized:
            err_payload = {
                "accepted": False,
                "error": (
                    "Your ranking contains 0 valid entries after "
                    "validation. chunk_ids must come from your saved "
                    "set or from this round's search results. Reasons "
                    f"must be ≥{cfg.min_reason_len} chars."
                ),
                "invalid_chunk_ids": invalid_entries[:20],
                "short_reason_ids": short_reasons[:20],
                "saved_sample": list(state.persistent_saved.keys())[:10],
            }
            err_payload["seen_this_round_sample"] = list(state.seen_this_round)[:10]
            return False, json.dumps(err_payload, ensure_ascii=False), []

        payload: dict[str, Any] = {
            "accepted": True,
            "n_ranked_submitted": len(ranking_arg),
            "n_ranked_valid": len(normalized),
        }
        payload["covered_concepts"] = covered_concepts
        payload["unresolved_concepts"] = unresolved_concepts
        state.final_coverage = {
            "covered_concepts": covered_concepts,
            "unresolved_concepts": unresolved_concepts,
        }
        if invalid_entries:
            payload["dropped_invalid_chunk_ids"] = invalid_entries[:20]
        if short_reasons:
            payload["dropped_short_reason_ids"] = short_reasons[:20]
        if len(normalized) < cfg.finalize_short_ranking_threshold:
            payload["warning"] = (
                "Very short ranking accepted, but multi-hop queries usually "
                "need 4-8 evidence documents — confirm this is intentional."
            )
        return True, json.dumps(payload, ensure_ascii=False), normalized
