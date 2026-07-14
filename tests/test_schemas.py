import json
import pathlib

from retriever_agent import AgentConfig
from retriever_agent.tools.loader import load_tool_schemas

_REF = json.loads((pathlib.Path(__file__).resolve().parent / "_schemas_reference.json").read_text())


def test_tool_schemas_match_reference():
    assert load_tool_schemas(AgentConfig()) == _REF
