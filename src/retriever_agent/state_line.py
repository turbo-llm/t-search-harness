from __future__ import annotations

from .config import AgentConfig
from .state import SessionState
from .types import ContextBudget


def render_state_line(
    budget: ContextBudget,
    state: SessionState,
    config: AgentConfig,
    *,
    hard_locked: bool,
    is_last_round: bool,
    max_rounds: int,
) -> str:
    """Render the bracketed status line appended to every tool result.

    Args:
        budget: The round token budget.
        state: The session state.
        config: The agent config (for the degenerate zero-new threshold).
        hard_locked: Whether search_corpus is currently locked.
        is_last_round: Whether this is the last round.
        max_rounds: The configured round ceiling.

    Returns:
        The status line, plus any budget, degenerate-state, or locked-search advisories.
    """
    pct = int((budget.current_tokens / max(budget.max_tokens, 1)) * 100)
    line = (
        f"[Round {state.current_round}/{max_rounds} | "
        f"Budget: {budget.current_tokens}/{budget.max_tokens} ({pct}%) | "
        f"Saved: {len(state.persistent_saved)} chunks | "
        f"Searches this round: {state.searches_this_round}]"
    )
    notes: list[str] = []
    if hard_locked:
        if is_last_round:
            notes.append(
                "[Budget at 75%+ in LAST round: search_corpus is locked. "
                "Call finalize_ranking (ranking + covered_concepts + "
                "unresolved_concepts) with your best candidates.]"
            )
        else:
            notes.append(
                "[Budget at 75%+: search_corpus is locked. Call "
                "save_and_advance (saved_chunks + drop_from_saved + "
                "covered_concepts + unresolved_concepts + "
                "round_summary + next_round_goal) to advance to the "
                "next round, OR finalize_ranking if evidence is "
                "complete.]"
            )
    elif pct >= 65:
        notes.append(
            "[Budget at 65%+. Prepare save_and_advance: identify "
            "which chunks are solid evidence, which concepts are "
            "still unresolved. search_corpus will lock at 75%.]"
        )
    elif pct >= 50:
        notes.append(
            "[Budget at 50%+. Start tracking coverage explicitly: "
            "are you close to having evidence for every concept? If "
            "yes, plan save_and_advance or finalize. If no, focus "
            "searches on uncovered concepts.]"
        )
    if state.consecutive_zero_new_this_round >= config.degenerate_zero_new_threshold:
        notes.append(
            f"[State warning: {state.consecutive_zero_new_this_round} "
            "consecutive searches returned no new chunks this round. "
            "Angles may be exhausted. Consider save_and_advance with "
            "a different angle, or finalize_ranking.]"
        )
    if state.locked_search_turns_streak >= 1:
        notes.append(
            "[You just called search_corpus while locked. Further "
            "locked-search turns will force-terminate the session. "
            "Use save_and_advance or finalize_ranking now.]"
        )
    if notes:
        return line + "\n" + "\n".join(notes)
    return line
