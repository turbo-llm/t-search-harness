from retriever_agent import AgentConfig, RetrieverAgent


class _StubLLM:
    def call(self, messages, tools):
        return None, None


class _StubSearch:
    def search(self, query, top_k):
        return "[]"


def test_retrieve_returns_result_on_llm_error():
    agent = RetrieverAgent(AgentConfig(model="x", max_rounds=2), _StubLLM(), _StubSearch())
    res = agent.retrieve("q")
    assert res.termination_reason == "llm_error"
    assert res.documents == []
    assert res.rounds_completed == 1
    assert set(res.tool_call_counts) == {
        "search",
        "search_turns",
        "save_and_advance",
        "finalize_ranking",
        "save_rejected",
        "finalize_rejected",
    }
