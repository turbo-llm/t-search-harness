from __future__ import annotations

import json
from typing import Any

from ..state import SavedChunk, SessionState
from .base import Tool, ToolContext, ToolResult


class SaveAdvanceTool(Tool):
    """save_and_advance: end the round and carry the saved set into the next round."""

    name = "save_and_advance"

    def gate(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult | None:
        """Reject save_and_advance on the last round; the session must end via finalize_ranking.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            A ToolResult to reject the call, or None to proceed to execute().
        """
        if ctx.is_last_round:
            payload = json.dumps(
                {"error": "this is the last round; save_and_advance is disabled. Call finalize_ranking."},
                ensure_ascii=False,
            )
            return ToolResult(payload=payload)
        return None

    def execute(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult:
        """Validate the save and, on success, mutate state and mark the round advanced.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            The ToolResult; ``pending_advance`` carries the parsed args when accepted.
        """
        accepted, payload, parsed = self._save_and_advance(args, state)
        if accepted:
            return ToolResult(payload=payload, accepted=True, pending_advance=parsed)
        return ToolResult(payload=payload)

    def _save_and_advance(self, args: dict[str, Any], state: SessionState) -> tuple[bool, str, dict[str, Any]]:
        """Validate the save arguments and, on success, mutate the saved set.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.

        Returns:
            ``(accepted, payload_json, parsed_args)``; ``parsed_args`` is empty on reject.
        """
        cfg = self._config
        saved_chunks_raw = args.get("saved_chunks") or []
        drop_from_saved_raw = args.get("drop_from_saved") or []
        round_summary = args.get("round_summary", "")
        next_round_goal = args.get("next_round_goal", "")
        covered_raw = args.get("covered_concepts")
        unresolved_raw = args.get("unresolved_concepts")

        def reject(msg: str, **extra: Any) -> tuple[bool, str, dict[str, Any]]:
            return (
                False,
                json.dumps({"accepted": False, "error": msg, **extra}, ensure_ascii=False),
                {},
            )

        if not isinstance(saved_chunks_raw, list):
            return reject("saved_chunks must be an array of {chunk_id, reason}.")
        if not isinstance(drop_from_saved_raw, list):
            return reject("drop_from_saved must be an array of chunk_ids.")
        if not isinstance(round_summary, str) or not isinstance(next_round_goal, str):
            return reject("round_summary and next_round_goal must be strings.")
        if not isinstance(covered_raw, list) or not isinstance(unresolved_raw, list):
            return reject(
                "covered_concepts and unresolved_concepts must be arrays of strings "
                "(one of them may be empty, but both must be present as arrays)."
            )

        covered_concepts = [str(x).strip() for x in covered_raw if str(x).strip()]
        unresolved_concepts = [str(x).strip() for x in unresolved_raw if str(x).strip()]

        if not covered_concepts and not unresolved_concepts:
            return reject(
                "At least one of covered_concepts or unresolved_concepts must be non-empty. "
                "Decompose your query into atomic concepts and declare which are supported vs. still open."
            )
        concept_overlap = set(covered_concepts) & set(unresolved_concepts)
        if concept_overlap:
            return reject(
                "covered_concepts and unresolved_concepts overlap: "
                f"{sorted(concept_overlap)}. A concept is either supported by evidence or still open — pick one."
            )

        drop_ids = [str(x) for x in drop_from_saved_raw if str(x)]
        normalized_saved: list[dict[str, Any]] = []
        for item in saved_chunks_raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not cid:
                continue
            normalized_saved.append({"chunk_id": cid, "reason": reason})

        if state.searches_this_round < cfg.min_searches_before_save:
            return reject(
                f"save_and_advance requires at least {cfg.min_searches_before_save} searches this round "
                f"(you made {state.searches_this_round})."
            )
        if len(normalized_saved) < cfg.min_saved_per_save_call:
            return reject(
                f"save_and_advance requires saving at least {cfg.min_saved_per_save_call} chunk. "
                f"If no chunk is worth saving after {cfg.min_searches_before_save}+ searches, call "
                "finalize_ranking with prior saved chunks, or keep searching."
            )
        if len(round_summary.strip()) < cfg.min_round_summary_len:
            return reject(
                f"round_summary too short (len={len(round_summary.strip())}, min={cfg.min_round_summary_len}). "
                "Describe what you covered, what's remaining, new leads, and what didn't work."
            )
        if len(next_round_goal.strip()) < cfg.min_next_goal_len:
            return reject(
                f"next_round_goal too short (len={len(next_round_goal.strip())}, min={cfg.min_next_goal_len}). "
                "State the concept to pursue and specific angles to try."
            )

        saved_ids = {item["chunk_id"] for item in normalized_saved}
        drop_ids_set = set(drop_ids)
        overlap = saved_ids & drop_ids_set
        if overlap:
            return reject(f"chunk_ids cannot appear in both saved_chunks and drop_from_saved: {sorted(overlap)}")
        for cid in drop_ids_set:
            if cid not in state.persistent_saved:
                return reject(
                    f"drop_from_saved references chunk_id '{cid}' which is not in your currently-saved set.",
                    current_saved_sample=list(state.persistent_saved.keys())[:10],
                )

        valid_source = state.seen_this_round | set(state.persistent_saved.keys())
        for item in normalized_saved:
            if len(item["reason"]) < cfg.min_reason_len:
                return reject(
                    f"reason for chunk_id '{item['chunk_id']}' too short "
                    f"(len={len(item['reason'])}, min={cfg.min_reason_len}). "
                    "Name the specific fact or entity this chunk contributes."
                )
            if item["chunk_id"] not in valid_source:
                return reject(
                    f"chunk_id '{item['chunk_id']}' is not in this round's search results or in your saved set. "
                    "You can only save chunks you actually received via search this round, or update reasons "
                    "for chunks already in your saved set.",
                    seen_this_round_sample=list(state.seen_this_round)[:10],
                    persistent_saved_sample=list(state.persistent_saved.keys())[:10],
                )

        for cid in drop_ids_set:
            state.persistent_saved.pop(cid, None)

        newly_added = 0
        updated_reason = 0
        for item in normalized_saved:
            cid = item["chunk_id"]
            reason = item["reason"]
            if cid in state.persistent_saved:
                state.persistent_saved[cid].reason = reason
                updated_reason += 1
            else:
                state.persistent_saved[cid] = SavedChunk(
                    chunk_id=cid,
                    snippet=state.chunk_snippet_cache.get(cid, ""),
                    reason=reason,
                    round_saved=state.current_round,
                    score=state.chunk_score_cache.get(cid, 0.0),
                )
                newly_added += 1

        parsed = {
            "round_summary": round_summary.strip(),
            "next_round_goal": next_round_goal.strip(),
            "covered_concepts": covered_concepts,
            "unresolved_concepts": unresolved_concepts,
            "goal_carried_from_prev_round": (
                state.round_summaries[-1]["next_round_goal"]
                if state.round_summaries
                else "initial decomposition of the query"
            ),
        }

        payload: dict[str, Any] = {
            "accepted": True,
            "round_closed": state.current_round,
            "next_round": state.current_round + 1,
            "saved_set_size_after": len(state.persistent_saved),
            "newly_added": newly_added,
            "updated_reason": updated_reason,
            "dropped": len(drop_ids_set),
        }
        if len(state.persistent_saved) > cfg.saved_soft_cap:
            payload["warning"] = (
                f"saved set has {len(state.persistent_saved)} chunks — over the soft cap of "
                f"{cfg.saved_soft_cap}. Consider using drop_from_saved in the next save_and_advance "
                "to trim redundant entries."
            )
        if state.round_summaries:
            prev_unresolved = set(state.round_summaries[-1].get("unresolved_concepts") or [])
            cur_unresolved = set(unresolved_concepts)
            if cur_unresolved and cur_unresolved == prev_unresolved:
                payload["advisory"] = (
                    "unresolved_concepts is unchanged from last round. If another round's search angles "
                    "wouldn't realistically close them, the remaining evidence may not be in the corpus — "
                    "consider finalize_ranking and declaring the residual concepts as honestly unresolved."
                )
        return True, json.dumps(payload, ensure_ascii=False), parsed
