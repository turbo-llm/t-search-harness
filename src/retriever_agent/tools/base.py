from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..clients.search import SearchClient
from ..config import AgentConfig
from ..state import SessionState


@dataclass
class ToolContext:
    """Read-only per-turn view passed to a tool's gate() and execute().

    Attributes:
        hard_locked: Whether the round budget is at or past the search lock.
        is_last_round: Whether this is the final round of the session.
        search: The injected search client (used by the search tool only).
    """

    hard_locked: bool
    is_last_round: bool
    search: SearchClient


@dataclass
class ToolResult:
    """Outcome of a single tool call.

    Attributes:
        payload: The JSON string returned to the model as the tool result (the
            harness appends a state line after it).
        accepted: Whether the call counts as progress: a search that reached its
            body, or a save/finalize that passed validation and mutated state.
        pending_advance: Parsed args when save_and_advance is accepted (else None).
        finalized_ranking: The normalized ranking when finalize_ranking is accepted.
        locked_search: Marker for a search rejected by the hard lock.
    """

    payload: str
    accepted: bool = False
    pending_advance: dict[str, Any] | None = None
    finalized_ranking: list[dict[str, Any]] = field(default_factory=list)
    locked_search: bool = False


class Tool:
    """Base class for the three retrieval tools.

    Each tool receives its numeric config and its OpenAI function schema at
    construction, so nothing is computed at import time. gate() runs pre-call
    rejects; execute() runs the accepted-path body. All mutation flows through
    the shared :class:`SessionState`.
    """

    name: str

    def __init__(self, config: AgentConfig, tool_schema: dict[str, Any]) -> None:
        self._config = config
        self._tool_schema = tool_schema

    def get_openai_tool_schema(self) -> dict[str, Any]:
        """Return the tool schema.

        Returns:
            The OpenAI function tool schema for this tool.
        """
        return self._tool_schema

    def gate(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult | None:
        """Run a pre-call reject check.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            A ToolResult to reject the call, or None to proceed to execute().
        """
        return None

    def execute(self, args: dict[str, Any], state: SessionState, ctx: ToolContext) -> ToolResult:
        """Run the tool body on arguments that passed the gate.

        Args:
            args: The parsed tool-call arguments.
            state: The shared session state.
            ctx: The read-only per-turn context.

        Returns:
            The ToolResult carrying the model-visible payload and any state markers.
        """
        raise NotImplementedError
