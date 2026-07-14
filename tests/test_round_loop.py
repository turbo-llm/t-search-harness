import json
import types

from retriever_agent import AgentConfig, RetrieverAgent


def _tc(call_id, name, args):
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
    return types.SimpleNamespace(id=call_id, type="function", function=fn)


def _msg(tool_calls):
    return types.SimpleNamespace(content="", tool_calls=tool_calls or None)


class _ScriptedLLM:
    """Replays a fixed sequence of assistant turns."""

    def __init__(self, turns):
        self._turns = list(turns)

    def call(self, messages, tools):
        return self._turns.pop(0), None


class _StubSearch:
    """Returns one unique hit per distinct query."""

    def search(self, query, top_k):
        return json.dumps([{"docid": f"doc-{query}", "snippet": f"text for {query}", "score": 0.9}])


def test_full_session_save_then_finalize():
    reason = "names the specific bridging entity"
    turns = [
        # round 1, turn 1: premature save -> rejected (needs 5 searches first)
        _msg(
            [
                _tc(
                    "c0",
                    "save_and_advance",
                    {
                        "saved_chunks": [{"chunk_id": "doc-q1", "reason": reason}],
                        "drop_from_saved": [],
                        "covered_concepts": ["a"],
                        "unresolved_concepts": ["b"],
                        "round_summary": "s" * 60,
                        "next_round_goal": "g" * 40,
                    },
                )
            ]
        ),
        # round 1, turn 2: five parallel searches
        _msg([_tc(f"c{i}", "search_corpus", {"query": f"q{i}"}) for i in range(1, 6)]),
        # round 1, turn 3: save doc-q1 -> round 2
        _msg(
            [
                _tc(
                    "c6",
                    "save_and_advance",
                    {
                        "saved_chunks": [{"chunk_id": "doc-q1", "reason": reason}],
                        "drop_from_saved": [],
                        "covered_concepts": ["a"],
                        "unresolved_concepts": ["b"],
                        "round_summary": "s" * 60,
                        "next_round_goal": "g" * 40,
                    },
                )
            ]
        ),
        # round 2 (last): finalize with the saved chunk
        _msg(
            [
                _tc(
                    "c7",
                    "finalize_ranking",
                    {
                        "ranking": [{"chunk_id": "doc-q1", "reason": reason}],
                        "covered_concepts": ["a", "b"],
                        "unresolved_concepts": [],
                    },
                )
            ]
        ),
    ]
    agent = RetrieverAgent(AgentConfig(model="x", max_rounds=2), _ScriptedLLM(turns), _StubSearch())
    res = agent.retrieve("test query")

    assert res.termination_reason == "finalized"
    assert res.rounds_completed == 2
    assert res.tool_call_counts["save_rejected"] == 1
    assert res.tool_call_counts["save_and_advance"] == 1
    assert res.tool_call_counts["search"] == 5
    assert res.tool_call_counts["finalize_ranking"] == 1
    assert [d.chunk_id for d in res.documents] == ["doc-q1"]
    assert res.documents[0].rank == 1
    assert res.documents[0].text == "text for q1"
    assert res.documents[0].retrieval_query == "q1"
    # a round with no searches is padded with the previous round's count
    assert res.searches_per_round == [5, 5]
    assert res.final_coverage_declaration == {"covered_concepts": ["a", "b"], "unresolved_concepts": []}
