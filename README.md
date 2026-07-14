# T-Search harness

The round-based agentic-search retriever harness for the **T-Search** model. A
session runs up to N rounds; each round starts with a fresh context and only chunks
the model explicitly saves via `save_and_advance` carry forward. Three tools drive
the loop — `search_corpus`, `save_and_advance`, `finalize_ranking` — and
`agent.retrieve(query)` returns a ranked `RetrievalResult`. The agent relies on an
LLM and a search client, and runs against any OpenAI-compatible endpoint and corpus
backend.

The harness is built for the **T-Search** model — [model card](https://huggingface.co/t-tech/T-Search).
Other models will run, but stable behavior is only expected with T-Search; conversely,
the prompts, tool schemas, and thresholds here are the operating point T-Search was
trained for — changing them changes the model's results.

## Installation

Requires Python ≥ 3.10. Distribution `t-search-harness`, import `retriever_agent`;
fully typed (ships `py.typed`). Dependency ranges are declared in `pyproject.toml`
and pinned in `poetry.lock`.

```bash
git clone https://github.com/turbo-llm/t-search-harness
cd t-search-harness
poetry install          # or: pip install -e .
```

## Usage

```python
from retriever_agent import AgentConfig, HttpSearchClient, OpenAILLMClient, RetrieverAgent

config = AgentConfig(model="t-tech/T-Search")
llm = OpenAILLMClient(["http://<vllm-host>:8000/v1"], config)
search = HttpSearchClient("http://<search-host>:8000")

agent = RetrieverAgent(config, llm, search)
result = agent.retrieve("your query")
for doc in result.documents:
    print(doc.rank, doc.doc_id, doc.score, doc.text)
```

`AgentConfig` is a `pydantic-settings` model — every field has a working default, so
`AgentConfig()` is valid; override per field (`AgentConfig(max_rounds=3)`), from a YAML
mapping (`AgentConfig.from_yaml(path)`), or from the environment
(`RETRIEVER_AGENT_<FIELD>`). `OpenAILLMClient` reads the model and sampling from the
config; the endpoints are passed separately since they are not an `AgentConfig` field.
`HttpSearchClient` is an example adapter over a generic HTTP backend — adapt its
request/response remap to your search server's shape (see below).

## Search backend contract

**You bring your own retrieval.** This repo ships the agent harness and search
contract. To run end-to-end you serve your own retrieval (vector DB + embedder, BM25,
your existing search service, …) over your own corpus and expose it through the one
method below.

The agent calls exactly one method on the injected search client:

- **`search(query: str, top_k: int) -> str`** — a JSON **string** encoding a **list** of objects; each item is read as `docid` (str; empty → skipped), `snippet` (str), `score` (float). `top_k` is the count the agent requested in its `search_corpus` tool call — the model can vary it per call. The harness does **not** truncate `snippet` — return snippets already bounded in length, or many large hits will overflow the model's context budget.

`search()` may raise: the harness attempts the call up to 3 times with backoff, then
surfaces the failure to the model as a tool error and the session continues. An empty
list is a valid result; malformed JSON is likewise surfaced to the model rather than
crashing the session.

Any object with that one method satisfies the contract (it's a structural `SearchClient` Protocol — no subclassing needed). The simplest path is an in-process wrapper around your retriever — no HTTP server required:

```python
import json

class MySearchClient:
    def __init__(self, index):
        self.index = index  # your retriever: vector DB, BM25, search service, …

    def search(self, query: str, top_k: int) -> str:
        hits = self.index.query(query, top_k)   # -> your retriever's hits
        return json.dumps([
            {"docid": h.id, "snippet": h.text, "score": h.score}
            for h in hits
        ])

search = MySearchClient(my_index)
agent = RetrieverAgent(config, llm, search)     # llm/config as in Usage above
```

If your retrieval already lives behind HTTP, use `HttpSearchClient` instead: point it at a backend taking `POST /search {"query","top_k"} -> {"result":[{"doc_id","snippet","score"}]}`, or adapt the remap to your backend's shape.

## Output contract

`agent.retrieve(query)` returns a `RetrievalResult`; the ranking is `result.documents`, a list of `RankedDocument`:

| field | meaning |
|-------|---------|
| `chunk_id` | the id of the retrieved chunk — the only id the harness carries |
| `doc_id` | alias of `chunk_id` (the harness has no separate document id) |
| `text` | the snippet the retriever saw during search |
| `score` | backend relevance score (passed through from your search client) |
| `rank` | 1-based position in the final ranking |
| `retrieval_query` | the query that first surfaced this chunk |

The retriever does **not** fetch full documents — `text` is the snippet. This is the fixed hand-off: take the ranking and feed it to whatever downstream you like (a generator, a re-ranker, your own full-text fetch).

`RetrievalResult` also carries per-run telemetry (`rounds_completed`, `tool_call_counts`, `round_summaries`, `searches_per_round`, …) and the raw transcript — `messages` / `all_round_messages` — for inspection. **Note:** the transcript embeds the full system prompt repeated per round, so read `result.documents` if you only want the ranking, and avoid blindly logging/publishing `.to_dict()` (it re-exposes the full harness prompt text).

## Hyperparameters

You are **not expected to change these** — the defaults are the tuned operating point.
Every knob is a field on `AgentConfig`; override it per field, from a YAML mapping, or
from the environment (`RETRIEVER_AGENT_<FIELD>`).

| field | default | what it controls |
|-------|---------|------------------|
| `model` | `None` | model name sent to the OpenAI-compatible endpoint |
| `temperature` / `top_p` | `0.7` / `1.0` | sampling per LLM turn |
| `max_tokens_per_turn` | `16384` | max generated tokens per assistant turn |
| `sampling_extra` | `{}` | extra sampling params (`top_k`, `min_p`, `repetition_penalty`, …) |
| `max_rounds` | `5` | max rounds per session (a ceiling, not a target) |
| `budget_tokens` | `32768` | per-round context token budget |
| `max_results` | `10` | max documents in the final ranking |
| `hard_lock_ratio` | `0.75` | budget fraction at which `search_corpus` locks (only save/finalize remain) |
| `min_searches_before_save` | `5` | `save_and_advance` is rejected before this many searches in the round |
| `min_saved_per_save_call` | `1` | minimum chunks `save_and_advance` must carry forward |
| `saved_soft_cap` | `15` | soft cap on the saved set (advisory only) |
| `min_reason_len` / `min_round_summary_len` / `min_next_goal_len` | `20` / `50` / `30` | minimum lengths for save/finalize text fields |
| `finalize_unresolved_reject_ratio` | `0.5` | reject finalize when this fraction or more of concepts is unresolved, unless last round |
| `finalize_short_ranking_threshold` | `3` | a final ranking shorter than this gets a "very short" warning |
| `degenerate_zero_new_threshold` | `5` | consecutive zero-new-chunk searches before a state warning |
| `free_text_guard_limit` | `2` | consecutive no-tool-call turns before force-terminate |
| `locked_search_streak_limit` | `2` | consecutive locked-search turns before force-terminate |
| `max_turns_per_round` | `60` | absolute safety cap on assistant turns per round |
| `llm_timeout_s` | `600.0` | OpenAI client request timeout, seconds |

The prompts (`src/retriever_agent/prompts.py`) and tool schemas
(`src/retriever_agent/tools/schemas.yaml`) *are* the harness — that text drives what
the model does. Changing it changes the model's behavior.

## Contacts

Maintainers: **anatolii.s.potapov@gmail.com**.

## License

Licensed under the [Apache License 2.0](LICENSE).
