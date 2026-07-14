import pathlib

from retriever_agent import prompts

_REF = (pathlib.Path(__file__).resolve().parent / "_system_prompt_reference.txt").read_text(encoding="utf-8")


def test_system_prompt_matches_reference():
    assert prompts.render_system_prompt(5) == _REF


def test_max_rounds_substituted():
    assert "up to 7 rounds" in prompts.render_system_prompt(7)
