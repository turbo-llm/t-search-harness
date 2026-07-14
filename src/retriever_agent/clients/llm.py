from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from ..config import AgentConfig

log = logging.getLogger(__name__)

_DEFAULT_MAX_TRIES = 5
_TOP_LEVEL_SAMPLING_KEYS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")
_EXTRA_BODY_SAMPLING_KEYS = ("top_k", "min_p", "repetition_penalty")


def extract_reasoning(message: Any) -> str | None:
    """Return the reasoning text a response message carries, if any.

    Args:
        message: The API response message object.

    Returns:
        The ``reasoning_content`` / ``reasoning`` value coerced to ``str``, or None.
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return str(val)
    return None


def wrap_inline_reasoning(reasoning: str | None, content: str) -> str:
    """Fold reasoning into assistant content as a ``<think>...</think>`` prefix.

    Args:
        reasoning: The reasoning text, if any.
        content: The assistant message content.

    Returns:
        The content with the reasoning folded in as an inline think block.
    """
    if reasoning and content:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>"
    return content


def _apply_sampling(create_kwargs: dict[str, Any], sampling_extra: dict[str, Any]) -> dict[str, Any]:
    """Merge extra sampling params into chat-completions kwargs.

    Top-level keys go on the request; extra-body keys go under ``extra_body``;
    unknown keys are dropped with a warning; None values are skipped.

    Args:
        create_kwargs: The chat-completions request kwargs, updated in place.
        sampling_extra: Extra sampling params to merge.

    Returns:
        The updated ``create_kwargs``.
    """
    for key, value in sampling_extra.items():
        if value is None:
            continue
        if key in _TOP_LEVEL_SAMPLING_KEYS:
            create_kwargs[key] = value
        elif key in _EXTRA_BODY_SAMPLING_KEYS:
            eb = create_kwargs.get("extra_body") or {}
            eb[key] = value
            create_kwargs["extra_body"] = eb
        else:
            log.warning("sampling_extra key %r is not recognized and was dropped", key)
    return create_kwargs


class LLMClient(Protocol):
    """The LLM interface the agent depends on."""

    def call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[Any | None, Any | None]:
        """Run one chat-completion turn.

        Args:
            messages: The conversation so far (OpenAI chat format).
            tools: The tool schemas available this turn.

        Returns:
            ``(message, usage)`` (``usage`` may be None) or ``(None, None)`` on
            failure. ``message`` exposes
            ``.content`` and ``.tool_calls`` (``.id`` / ``.type`` /
            ``.function.name`` / ``.function.arguments``).
        """
        ...


class OpenAILLMClient:
    """OpenAI-backed LLM client.

    Model and sampling come from the config. One endpoint is drawn at construction;
    construct a new client to re-draw. ``api_key`` defaults to a placeholder —
    self-hosted OpenAI-compatible servers ignore it; pass a real key for secured
    endpoints. ``openai_factory`` overrides the OpenAI constructor, e.g. for tests.
    """

    def __init__(
        self,
        endpoints: list[str],
        config: AgentConfig,
        *,
        api_key: str = "EMPTY",
        openai_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        factory = openai_factory if openai_factory is not None else OpenAI
        endpoint = random.choice(endpoints)
        self._client = factory(base_url=endpoint, api_key=api_key, timeout=config.llm_timeout_s)

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tries: int = _DEFAULT_MAX_TRIES,
    ) -> tuple[Any | None, Any | None]:
        """Call the LLM with retry.

        Args:
            messages: The conversation so far.
            tools: The tool schemas available this turn.
            max_tries: Max attempts on transient API errors.

        Returns:
            ``(message, usage)`` or ``(None, None)``.
        """
        cfg = self._config
        for attempt in range(max_tries):
            try:
                create_kwargs = dict(
                    model=cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens_per_turn,
                )
                _apply_sampling(create_kwargs, cfg.sampling_extra)
                resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[call-overload]
                return resp.choices[0].message, getattr(resp, "usage", None)
            except (APIError, APIConnectionError, APITimeoutError) as exc:
                wait = min(2**attempt + random.random(), 30)
                log.warning("LLM call attempt %d failed: %s – retrying in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001 — any other error ends the call cleanly
                log.error("Unexpected LLM error: %s", exc)
                return None, None
        log.error("All LLM retries exhausted")
        return None, None
