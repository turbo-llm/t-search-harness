from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """All tunable knobs for a retrieval session.

    Every field has a working default, so ``AgentConfig()`` is a valid config.
    Override in three ways: per field (``AgentConfig(max_rounds=3)``), from a YAML
    mapping (``AgentConfig.from_yaml(path)``), or from the environment — each field
    maps to ``RETRIEVER_AGENT_<FIELD>`` (e.g. ``RETRIEVER_AGENT_MAX_ROUNDS=3``);
    explicit arguments take precedence over the environment.

    You are not expected to change these; the defaults are the tuned operating point.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVER_AGENT_", extra="ignore", protected_namespaces=())

    # LLM sampling
    model: str | None = Field(default=None, description="Model name sent to the OpenAI-compatible endpoint.")
    temperature: float = Field(default=0.7, description="Sampling temperature per LLM turn.")
    top_p: float = Field(default=1.0, description="Nucleus sampling cutoff.")
    max_tokens_per_turn: int = Field(default=16384, description="Max generated tokens per assistant turn.")
    sampling_extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra sampling params merged into the request "
        "(top_k / min_p / repetition_penalty / presence_penalty / frequency_penalty).",
    )
    # round flow
    max_rounds: int = Field(default=5, description="Max rounds per session (a ceiling, not a target).")
    budget_tokens: int = Field(default=32768, description="Per-round context token budget.")
    max_results: int = Field(default=10, description="Max documents in the final ranking.")
    hard_lock_ratio: float = Field(
        default=0.75,
        description="Budget fraction at which search_corpus locks (only save/finalize remain).",
    )

    # gates
    min_searches_before_save: int = Field(
        default=5,
        description="save_and_advance is rejected before this many searches in the round.",
    )
    min_saved_per_save_call: int = Field(
        default=1, description="save_and_advance must carry at least this many chunks forward."
    )
    saved_soft_cap: int = Field(
        default=15,
        description="Soft cap on the persistent saved set (advisory only, never blocks).",
    )
    min_reason_len: int = Field(default=20, description="Min chars for a save/finalize chunk reason.")
    min_round_summary_len: int = Field(default=50, description="Min chars for the save_and_advance round_summary.")
    min_next_goal_len: int = Field(default=30, description="Min chars for the save_and_advance next_round_goal.")
    finalize_unresolved_reject_ratio: float = Field(
        default=0.5,
        description="Reject finalize when this fraction or more of declared concepts "
        "is unresolved, unless it is the last round.",
    )
    finalize_short_ranking_threshold: int = Field(
        default=3, description="A final ranking shorter than this gets a 'very short' warning."
    )

    # degenerate-state / termination guards
    degenerate_zero_new_threshold: int = Field(
        default=5, description="Consecutive zero-new-chunk searches before a state warning."
    )
    free_text_guard_limit: int = Field(
        default=2, description="Consecutive no-tool-call assistant turns before force-terminate."
    )
    locked_search_streak_limit: int = Field(
        default=2, description="Consecutive locked-search turns before force-terminate."
    )
    max_turns_per_round: int = Field(
        default=60,
        description="Absolute safety cap on assistant turns per round (budget usually bites first).",
    )

    # LLM client
    llm_timeout_s: float = Field(default=600.0, description="OpenAI client request timeout, seconds.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        """Load an ``AgentConfig`` from a YAML mapping.

        Args:
            path: Path to a YAML file holding a flat mapping of field names to values.

        Returns:
            The parsed config. Keys that are not fields are ignored.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
