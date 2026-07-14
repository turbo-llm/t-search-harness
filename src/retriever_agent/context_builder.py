from __future__ import annotations

from typing import Any

from .prompts import render_round_n_reentry, render_save_note, render_system_prompt
from .state import SessionState


def build_initial_messages(query: str, max_rounds: int) -> list[dict[str, Any]]:
    """Build the round-1 message list.

    Args:
        query: The user query.
        max_rounds: The configured round ceiling.

    Returns:
        The system + user messages that open the session.
    """
    return [
        {"role": "system", "content": render_system_prompt(max_rounds)},
        {"role": "user", "content": f"Query: {query}"},
    ]


def build_round_n_messages(query: str, state: SessionState, round_goal: str, max_rounds: int) -> list[dict[str, Any]]:
    """Rebuild context for round N ≥ 2 with the saved chunks and round history.

    Args:
        query: The user query.
        state: The session state carrying saved chunks and round summaries.
        round_goal: The goal carried from the previous round.
        max_rounds: The configured round ceiling.

    Returns:
        The fresh system + user messages for the new round.
    """
    if state.persistent_saved:
        saved_lines: list[str] = []
        for sc in state.persistent_saved.values():
            saved_lines.append(
                f"## chunk_id: {sc.chunk_id} (saved in round {sc.round_saved})\n"
                f"Reason: {sc.reason}\n"
                f"Snippet:\n{sc.snippet}\n\n---"
            )
        saved_block = "\n".join(saved_lines)
    else:
        saved_block = "_(no saved chunks yet)_"

    history_lines: list[str] = []
    overall_covered: list[str] = []
    overall_seen_concepts: list[str] = []
    for rs in state.round_summaries:
        goal_text = rs.get("goal") or "initial decomposition of the query"
        covered = rs.get("covered_concepts", []) or []
        unresolved = rs.get("unresolved_concepts", []) or []
        for c in covered:
            if c not in overall_covered:
                overall_covered.append(c)
        for c in covered + unresolved:
            if c not in overall_seen_concepts:
                overall_seen_concepts.append(c)
        history_lines.append(
            f"## Round {rs['round']}\n"
            f"Goal: {goal_text}\n"
            f"Covered concepts (declared in save): {covered}\n"
            f"Unresolved concepts (declared in save): {unresolved}\n"
            f"Summary:\n{rs['summary']}"
        )
    history_block = "\n\n".join(history_lines) if history_lines else "_(empty)_"

    still_unresolved = [c for c in overall_seen_concepts if c not in overall_covered]
    coverage_state = f"Covered overall: {overall_covered}\nStill unresolved: {still_unresolved}"

    save_tool_note = render_save_note(max_rounds, enabled=state.current_round < max_rounds)

    user_content = render_round_n_reentry(
        original_query=query,
        saved_block=saved_block,
        history_block=history_block,
        coverage_state=coverage_state,
        current_round=state.current_round,
        max_rounds=max_rounds,
        prev_round=state.current_round - 1,
        round_goal=round_goal or "_(no explicit goal carried)_",
        save_tool_note=save_tool_note,
    )

    return [
        {"role": "system", "content": render_system_prompt(max_rounds)},
        {"role": "user", "content": user_content},
    ]
