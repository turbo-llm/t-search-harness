from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol


class SearchClient(Protocol):
    """The search interface the agent depends on."""

    def search(self, query: str, top_k: int) -> str:
        """Search the corpus and return hits as a JSON string.

        Args:
            query: The search query.
            top_k: The number of hits the agent requested in its tool call.

        Returns:
            A JSON string encoding a list of ``{docid, snippet, score}`` objects.
            The agent drops empty-``docid`` items and does not truncate snippets,
            so the backend must return bounded snippets or large hits overflow
            the budget.
        """
        ...


class HttpSearchClient:
    """Example adapter for a generic HTTP search backend (urllib, no extra deps).

    ``search(query, top_k)`` returns the shape the agent expects: a JSON string of
    a list of ``{docid, snippet, score}``. Assumes a backend with
    ``POST /search {"query","top_k"} -> {"result":[{"doc_id","snippet","score"}]}``;
    adapt the remap to your backend.
    """

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post_json(self, path: str, body: dict[str, Any]) -> Any:
        """POST ``body`` as JSON and decode the JSON response.

        Args:
            path: The URL path appended to the base URL.
            body: The request payload.

        Returns:
            The decoded JSON response.
        """
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def search(self, query: str, top_k: int = 5) -> str:
        """Query the backend and remap its response to the agent's contract.

        Args:
            query: The search query.
            top_k: The number of hits to request.

        Returns:
            A JSON string of ``{docid, snippet, score}`` objects.
        """
        body = self._post_json("/search", {"query": query, "top_k": top_k})
        return json.dumps(
            [
                {
                    "docid": x.get("doc_id", ""),
                    "snippet": x.get("snippet", ""),
                    "score": x.get("score", 0.0),
                }
                for x in body.get("result", [])
            ]
        )
