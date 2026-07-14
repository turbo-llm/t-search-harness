from __future__ import annotations

import json
import logging
from typing import Any

from .clients.llm import LLMClient, extract_reasoning, wrap_inline_reasoning
from .clients.search import SearchClient
from .config import AgentConfig
from .context_builder import build_initial_messages, build_round_n_messages
from .final_ranking import build_final_ranking
from .state import SessionState
from .state_line import render_state_line
from .tools.base import ToolContext
from .tools.loader import build_tools
from .types import ContextBudget, RetrievalResult

log = logging.getLogger(__name__)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate the token count of ``messages`` by characters / 4.

    Used for the pre-call budget estimate; the real API ``usage`` count replaces it
    after each turn when the endpoint reports usage.

    Args:
        messages: The conversation messages.

    Returns:
        The character-based token estimate.
    """
    return len(json.dumps(messages, ensure_ascii=False)) // 4


class RetrieverAgent:
    """Round-based retriever with a fresh context per round.

    Thread-safe: all mutable state lives inside ``retrieve()``.
    """

    def __init__(self, config: AgentConfig, llm: LLMClient, search: SearchClient) -> None:
        self._config = config
        self._llm = llm
        self._search = search
        self._tools = build_tools(config)
        defaults = AgentConfig.model_construct().model_dump()
        non_default = {k: v for k, v in config.model_dump().items() if v != defaults.get(k)}
        if non_default:
            log.info("non-default config: %s", non_default)

    def retrieve(
        self,
        query: str,
        max_results: int | None = None,
        budget_tokens: int | None = None,
    ) -> RetrievalResult:
        """Run the round-based retrieval loop and return ranked documents.

        Args:
            query: The user query.
            max_results: Max documents in the final ranking (defaults to the config value).
            budget_tokens: Per-round token budget (defaults to the config value).

        Returns:
            The RetrievalResult carrying the ranking and per-run telemetry.
        """
        cfg = self._config
        max_rounds = cfg.max_rounds
        max_results = cfg.max_results if max_results is None else max_results
        budget_tokens = cfg.budget_tokens if budget_tokens is None else budget_tokens

        state = SessionState()
        budget = ContextBudget(max_tokens=budget_tokens, hard_lock_ratio=cfg.hard_lock_ratio)
        stats = {
            "search": 0,
            "search_turns": 0,
            "save_and_advance": 0,
            "finalize": 0,
            "finalize_rejected": 0,
            "save_rejected": 0,
        }
        all_round_messages: list[list[dict[str, Any]]] = []

        llm = self._llm
        search_schema = self._tools["search_corpus"].get_openai_tool_schema()
        save_schema = self._tools["save_and_advance"].get_openai_tool_schema()
        finalize_schema = self._tools["finalize_ranking"].get_openai_tool_schema()
        last_round_searches_snapshot = 0

        messages: list[dict[str, Any]] = build_initial_messages(query, max_rounds)

        termination_reason = "max_rounds_reached"
        finalized_ranking: list[dict[str, Any]] = []
        consecutive_empty_assistant = 0
        turns_this_round = 0

        while state.current_round <= max_rounds:
            budget.current_tokens = _estimate_tokens(messages)
            hard_locked = budget.hard_locked
            is_last_round = state.current_round >= max_rounds

            if turns_this_round >= cfg.max_turns_per_round:
                log.warning("round=%d hit turn cap (%d); terminating", state.current_round, cfg.max_turns_per_round)
                termination_reason = "max_rounds_reached"
                break

            tools = [search_schema]
            if not is_last_round:
                tools.append(save_schema)
            tools.append(finalize_schema)
            allowed_tool_names = {t["function"]["name"] for t in tools}

            log.info(
                "round=%d turn=%d tokens~%d/%d saved=%d searches_this_round=%d hard_locked=%s last_round=%s",
                state.current_round,
                turns_this_round,
                budget.current_tokens,
                budget.max_tokens,
                len(state.persistent_saved),
                state.searches_this_round,
                hard_locked,
                is_last_round,
            )

            message, usage = llm.call(messages, tools=tools)
            if message is None:
                termination_reason = "llm_error"
                break
            turns_this_round += 1

            if usage and hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
                budget.current_tokens = (usage.prompt_tokens or 0) + (getattr(usage, "completion_tokens", 0) or 0)

            content = message.content or ""
            reasoning = extract_reasoning(message)
            inline_content = wrap_inline_reasoning(reasoning, content)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": inline_content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if not message.tool_calls:
                consecutive_empty_assistant += 1
                log.warning(
                    "round=%d empty assistant (no tool_calls), streak=%d",
                    state.current_round,
                    consecutive_empty_assistant,
                )
                if consecutive_empty_assistant >= cfg.free_text_guard_limit:
                    termination_reason = "free_text_loop"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Every turn must contain tool calls. Your last "
                            "message had no tool call. Make a tool call now: "
                            "search_corpus, save_and_advance, or "
                            "finalize_ranking."
                        ),
                    }
                )
                continue
            consecutive_empty_assistant = 0

            pending_advance: dict[str, Any] | None = None
            pending_finalize = False
            turn_had_progress_call = False
            turn_had_locked_search = False
            turn_had_successful_search = False

            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if name not in allowed_tool_names:
                    result_str = json.dumps(
                        {
                            "error": (
                                f"Tool '{name}' is not available in this turn. "
                                f"Allowed tools: {sorted(allowed_tool_names)}."
                            ),
                        },
                        ensure_ascii=False,
                    )
                elif pending_advance is not None:
                    result_str = json.dumps(
                        {
                            "error": (
                                "round already advanced by an earlier tool call in this turn; this call is ignored."
                            ),
                        },
                        ensure_ascii=False,
                    )
                elif pending_finalize:
                    result_str = json.dumps(
                        {
                            "error": (
                                "session already ended by finalize_ranking earlier in this turn; this call is ignored."
                            ),
                        },
                        ensure_ascii=False,
                    )
                elif name in self._tools:
                    tool = self._tools[name]
                    ctx = ToolContext(hard_locked=hard_locked, is_last_round=is_last_round, search=self._search)
                    gated = tool.gate(args, state, ctx)
                    res = gated if gated is not None else tool.execute(args, state, ctx)
                    result_str = res.payload
                    if name == "search_corpus":
                        if res.locked_search:
                            turn_had_locked_search = True
                        elif res.accepted:
                            stats["search"] += 1
                            turn_had_progress_call = True
                            turn_had_successful_search = True
                    elif name == "save_and_advance":
                        if res.accepted:
                            pending_advance = res.pending_advance
                            stats["save_and_advance"] += 1
                            turn_had_progress_call = True
                        else:
                            stats["save_rejected"] += 1
                    elif name == "finalize_ranking":
                        if res.accepted:
                            finalized_ranking = res.finalized_ranking
                            pending_finalize = True
                            stats["finalize"] += 1
                            turn_had_progress_call = True
                        else:
                            stats["finalize_rejected"] += 1
                            state.finalize_reject_count += 1
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)

                state_line = render_state_line(
                    budget,
                    state,
                    cfg,
                    hard_locked=hard_locked,
                    is_last_round=is_last_round,
                    max_rounds=max_rounds,
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"{result_str}\n\n{state_line}"})

            if turn_had_successful_search:
                stats["search_turns"] += 1

            if turn_had_progress_call:
                state.locked_search_turns_streak = 0
            elif turn_had_locked_search:
                state.locked_search_turns_streak += 1
                if state.locked_search_turns_streak >= cfg.locked_search_streak_limit:
                    log.warning("round=%d locked-search streak hit limit; terminating", state.current_round)
                    termination_reason = "search_lock_loop"
                    break

            if pending_finalize:
                termination_reason = "finalized"
                break

            if pending_advance is not None:
                state.round_summaries.append(
                    {
                        "round": state.current_round,
                        "goal": pending_advance.get("goal_carried_from_prev_round", ""),
                        "summary": pending_advance["round_summary"],
                        "covered_concepts": pending_advance.get("covered_concepts", []),
                        "unresolved_concepts": pending_advance.get("unresolved_concepts", []),
                        "next_round_goal": pending_advance["next_round_goal"],
                    }
                )
                state.searches_per_round.append(state.searches_this_round)
                last_round_searches_snapshot = state.searches_this_round
                all_round_messages.append(list(messages))
                state.current_round += 1
                state.seen_this_round = set()
                state.searches_this_round = 0
                state.consecutive_zero_new_this_round = 0
                turns_this_round = 0

                if state.current_round > max_rounds:
                    termination_reason = "max_rounds_reached"
                    break

                messages = build_round_n_messages(query, state, pending_advance["next_round_goal"], max_rounds)
                continue

        trailing_searches = state.searches_this_round or last_round_searches_snapshot
        while len(state.searches_per_round) < state.current_round:
            state.searches_per_round.append(trailing_searches)

        all_round_messages.append(list(messages))

        documents = build_final_ranking(finalized_ranking, state, max_results)

        persistent_saved_final = [
            {
                "chunk_id": sc.chunk_id,
                "reason": sc.reason,
                "round_saved": sc.round_saved,
                "score": sc.score,
            }
            for sc in state.persistent_saved.values()
        ]

        rounds_completed = (
            state.current_round if termination_reason == "finalized" else min(state.current_round, max_rounds)
        )
        log.info(
            "retrieve done: termination=%s rounds_completed=%d search_calls=%d search_turns=%d "
            "save_and_advance=%d finalize=%d save_rejected=%d finalize_rejected=%d "
            "saved_final=%d search_duplicates=%d",
            termination_reason,
            rounds_completed,
            stats["search"],
            stats["search_turns"],
            stats["save_and_advance"],
            stats["finalize"],
            stats["save_rejected"],
            stats["finalize_rejected"],
            len(state.persistent_saved),
            state.search_duplicate_count,
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            total_search_calls=stats["search"],
            total_search_turns=stats["search_turns"],
            total_finalize_calls=stats["finalize"],
            rounds_completed=rounds_completed,
            termination_reason=termination_reason,
            save_and_advance_count=stats["save_and_advance"],
            messages=messages,
            all_round_messages=all_round_messages,
            persistent_saved_final=persistent_saved_final,
            round_summaries=state.round_summaries,
            searches_per_round=state.searches_per_round,
            tool_call_counts={
                "search": stats["search"],
                "search_turns": stats["search_turns"],
                "save_and_advance": stats["save_and_advance"],
                "finalize_ranking": stats["finalize"],
                "save_rejected": stats["save_rejected"],
                "finalize_rejected": stats["finalize_rejected"],
            },
            retrieved_docids=sorted(state.global_seen_chunk_ids),
            final_coverage_declaration=dict(state.final_coverage),
            finalize_reject_count=state.finalize_reject_count,
            search_duplicate_count=state.search_duplicate_count,
        )
