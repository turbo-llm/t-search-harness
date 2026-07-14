from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ..config import AgentConfig
from .base import Tool
from .finalize import FinalizeTool
from .save_and_advance import SaveAdvanceTool
from .search import SearchTool

_SCHEMAS_PATH = Path(__file__).with_name("schemas.yaml")
_TOOL_CLASSES: dict[str, type[Tool]] = {
    "search_corpus": SearchTool,
    "save_and_advance": SaveAdvanceTool,
    "finalize_ranking": FinalizeTool,
}


def _substitute(obj: Any, values: dict[str, str]) -> Any:
    """Replace ``__KEY__`` tokens in every string of a nested structure.

    Args:
        obj: The structure to walk (str / dict / list; other types pass through).
        values: Token name to replacement value.

    Returns:
        The structure with all tokens replaced.
    """
    if isinstance(obj, str):
        for key, value in values.items():
            token = f"__{key}__"
            if token in obj:
                obj = obj.replace(token, value)
        return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, values) for v in obj]
    return obj


def load_tool_schemas(config: AgentConfig, path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the tool schemas and substitute ``__KEY__`` tokens from the config.

    Args:
        config: The agent config whose scalar (str/int/float) fields fill the
            schema ``__KEY__`` tokens.
        path: Optional override for the schemas YAML (defaults to the packaged file).

    Returns:
        A mapping of tool name to its OpenAI function schema.
    """
    raw = yaml.safe_load(Path(path or _SCHEMAS_PATH).read_text(encoding="utf-8"))
    values = {
        key.upper(): str(value) for key, value in config.model_dump().items() if isinstance(value, (int, float, str))
    }
    return {name: _substitute(copy.deepcopy(schema), values) for name, schema in raw.items()}


def build_tools(config: AgentConfig, path: str | Path | None = None) -> dict[str, Tool]:
    """Build the three retrieval tools from the config and the tool schemas.

    Args:
        config: The agent config passed to each tool instance.
        path: Optional override for the schemas YAML.

    Returns:
        A mapping of tool name to its constructed Tool instance.
    """
    schemas = load_tool_schemas(config, path)
    return {name: cls(config, schemas[name]) for name, cls in _TOOL_CLASSES.items()}
