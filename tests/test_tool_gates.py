import json

from retriever_agent import AgentConfig
from retriever_agent.state import SessionState
from retriever_agent.tools.base import ToolContext
from retriever_agent.tools.loader import build_tools

TOOLS = build_tools(AgentConfig())
REASON = "names the specific bridging entity"
SAVE_ARGS = {
    "saved_chunks": [{"chunk_id": "c1", "reason": REASON}],
    "drop_from_saved": [],
    "covered_concepts": ["a"],
    "unresolved_concepts": ["b"],
    "round_summary": "s" * 60,
    "next_round_goal": "g" * 40,
}


def _ctx(hard_locked=False, is_last_round=False, search=None):
    return ToolContext(hard_locked=hard_locked, is_last_round=is_last_round, search=search)


def _payload(res):
    return json.loads(res.payload)


class _DictSearch:
    def search(self, query, top_k):
        return json.dumps({"error": "index loading"})


# ── search_corpus ────────────────────────────────────────────────────────


def test_search_hard_lock_message():
    res = TOOLS["search_corpus"].gate({"query": "x"}, SessionState(), _ctx(hard_locked=True))
    assert res is not None
    assert res.locked_search
    p = _payload(res)
    assert p["error"] == (
        "Round budget at 75%+. search_corpus is locked for this round. "
        "Required: call save_and_advance (saved_chunks, drop_from_saved, "
        "covered_concepts, unresolved_concepts, round_summary, next_round_goal) "
        "to advance to the next round with fresh context, OR call finalize_ranking "
        "(ranking, covered_concepts, unresolved_concepts) if current evidence is sufficient."
    )
    assert p["is_last_round"] is False
    assert p["saved_set_size"] == 0


def test_search_duplicate_query_message():
    st = SessionState()
    st.search_history.add("hello world")
    st.search_history_round["hello world"] = 1
    res = TOOLS["search_corpus"].gate({"query": "Hello   World"}, st, _ctx())
    p = _payload(res)
    assert p["error"] == "duplicate_query"
    assert p["note"] == (
        "Query 'Hello   World' was already issued in round 1. Vary terms, paraphrase, or pivot to a different angle."
    )
    assert st.search_duplicate_count == 1
    assert st.consecutive_zero_new_this_round == 1


def test_search_empty_query():
    res = TOOLS["search_corpus"].execute({"query": "  "}, SessionState(), _ctx())
    assert _payload(res) == {"error": "Empty query."}


def test_search_non_list_backend_response():
    res = TOOLS["search_corpus"].execute({"query": "x"}, SessionState(), _ctx(search=_DictSearch()))
    assert _payload(res) == {"error": "Backend search returned invalid JSON."}


# ── save_and_advance ─────────────────────────────────────────────────────


def test_save_disabled_on_last_round():
    res = TOOLS["save_and_advance"].gate({}, SessionState(), _ctx(is_last_round=True))
    assert _payload(res)["error"] == ("this is the last round; save_and_advance is disabled. Call finalize_ranking.")


def test_save_requires_min_searches():
    res = TOOLS["save_and_advance"].execute(dict(SAVE_ARGS), SessionState(), _ctx())
    assert _payload(res)["error"] == "save_and_advance requires at least 5 searches this round (you made 0)."


def test_save_requires_min_saved_chunks():
    st = SessionState()
    st.searches_this_round = 5
    res = TOOLS["save_and_advance"].execute({**SAVE_ARGS, "saved_chunks": []}, st, _ctx())
    assert _payload(res)["error"] == (
        "save_and_advance requires saving at least 1 chunk. If no chunk is worth "
        "saving after 5+ searches, call finalize_ranking with prior saved chunks, "
        "or keep searching."
    )


def test_save_short_round_summary():
    st = SessionState()
    st.searches_this_round = 5
    st.seen_this_round.add("c1")
    res = TOOLS["save_and_advance"].execute({**SAVE_ARGS, "round_summary": "sss"}, st, _ctx())
    assert _payload(res)["error"] == (
        "round_summary too short (len=3, min=50). Describe what you covered, "
        "what's remaining, new leads, and what didn't work."
    )


def test_save_short_next_round_goal():
    st = SessionState()
    st.searches_this_round = 5
    st.seen_this_round.add("c1")
    res = TOOLS["save_and_advance"].execute({**SAVE_ARGS, "next_round_goal": "ggg"}, st, _ctx())
    assert _payload(res)["error"] == (
        "next_round_goal too short (len=3, min=30). State the concept to pursue and specific angles to try."
    )


def test_save_short_reason():
    st = SessionState()
    st.searches_this_round = 5
    st.seen_this_round.add("c1")
    args = {**SAVE_ARGS, "saved_chunks": [{"chunk_id": "c1", "reason": "ab"}]}
    res = TOOLS["save_and_advance"].execute(args, st, _ctx())
    assert _payload(res)["error"] == (
        "reason for chunk_id 'c1' too short (len=2, min=20). Name the specific fact or entity this chunk contributes."
    )


def test_save_unknown_chunk_source():
    st = SessionState()
    st.searches_this_round = 5
    res = TOOLS["save_and_advance"].execute(dict(SAVE_ARGS), st, _ctx())
    assert _payload(res)["error"] == (
        "chunk_id 'c1' is not in this round's search results or in your saved set. "
        "You can only save chunks you actually received via search this round, or "
        "update reasons for chunks already in your saved set."
    )


def test_save_concept_overlap():
    st = SessionState()
    st.searches_this_round = 5
    st.seen_this_round.add("c1")
    args = {**SAVE_ARGS, "covered_concepts": ["a"], "unresolved_concepts": ["a"]}
    res = TOOLS["save_and_advance"].execute(args, st, _ctx())
    assert _payload(res)["error"] == (
        "covered_concepts and unresolved_concepts overlap: ['a']. "
        "A concept is either supported by evidence or still open — pick one."
    )


def test_save_accepted_payload():
    st = SessionState()
    st.searches_this_round = 5
    st.seen_this_round.add("c1")
    res = TOOLS["save_and_advance"].execute(dict(SAVE_ARGS), st, _ctx())
    p = _payload(res)
    assert res.accepted
    assert p["accepted"] is True
    assert p["round_closed"] == 1
    assert p["next_round"] == 2
    assert p["saved_set_size_after"] == 1
    assert p["newly_added"] == 1
    assert "c1" in st.persistent_saved


# ── finalize_ranking ─────────────────────────────────────────────────────


def _finalize(args, st=None, is_last_round=False):
    return TOOLS["finalize_ranking"].execute(args, st or SessionState(), _ctx(is_last_round=is_last_round))


def test_finalize_empty_ranking():
    res = _finalize({"ranking": [], "covered_concepts": ["a"], "unresolved_concepts": []})
    assert _payload(res)["error"] == (
        "Empty ranking rejected. Submit your best candidates from your current "
        "context (saved chunks + this round's seen)."
    )


def test_finalize_missing_concept_arrays():
    st = SessionState()
    st.seen_this_round.add("c1")
    res = _finalize({"ranking": [{"chunk_id": "c1", "reason": REASON}]}, st)
    assert _payload(res)["error"] == (
        "covered_concepts and unresolved_concepts are required (arrays of strings). "
        "Declare which atoms of your query decomposition are supported by the final "
        "ranking and which are still open."
    )


def test_finalize_concept_overlap():
    st = SessionState()
    st.seen_this_round.add("c1")
    args = {
        "ranking": [{"chunk_id": "c1", "reason": REASON}],
        "covered_concepts": ["a"],
        "unresolved_concepts": ["a"],
    }
    res = _finalize(args, st)
    assert _payload(res)["error"] == (
        "covered_concepts and unresolved_concepts overlap: ['a']. Pick one bucket per concept."
    )


def test_finalize_unresolved_majority_rejected():
    st = SessionState()
    st.seen_this_round.add("c1")
    args = {
        "ranking": [{"chunk_id": "c1", "reason": REASON}],
        "covered_concepts": ["a"],
        "unresolved_concepts": ["b"],
    }
    res = _finalize(args, st)
    assert _payload(res)["error"] == (
        "finalize_ranking rejected: 1/2 concepts still unresolved (50%). "
        "Finalize is for when the query is actually covered. Call save_and_advance "
        "instead to continue in a fresh round targeting the unresolved concepts."
    )


def test_finalize_unresolved_majority_allowed_on_last_round():
    st = SessionState()
    st.seen_this_round.add("c1")
    args = {
        "ranking": [{"chunk_id": "c1", "reason": REASON}],
        "covered_concepts": ["a"],
        "unresolved_concepts": ["b"],
    }
    res = _finalize(args, st, is_last_round=True)
    p = _payload(res)
    assert res.accepted
    assert p["accepted"] is True
    assert p["warning"] == (
        "Very short ranking accepted, but multi-hop queries usually need 4-8 "
        "evidence documents — confirm this is intentional."
    )


def test_finalize_zero_valid_entries():
    st = SessionState()
    st.seen_this_round.add("c1")
    args = {
        "ranking": [{"chunk_id": "c1", "reason": "ab"}],
        "covered_concepts": ["a", "b"],
        "unresolved_concepts": [],
    }
    res = _finalize(args, st)
    assert _payload(res)["error"] == (
        "Your ranking contains 0 valid entries after validation. chunk_ids must "
        "come from your saved set or from this round's search results. Reasons "
        "must be ≥20 chars."
    )
