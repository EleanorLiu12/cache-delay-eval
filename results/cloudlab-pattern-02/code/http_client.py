"""Small dependency-free client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class VLLMClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _request(self, path: str, payload: dict[str, Any] | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        return urllib.request.urlopen(request, timeout=self.timeout_s)

    def json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        with self._request(path, payload) as response:
            return json.loads(response.read())

    def health(self) -> None:
        with self._request("/health") as response:
            if response.status >= 300:
                raise RuntimeError(f"health endpoint returned HTTP {response.status}")

    def tokenize(self, prompt: str) -> list[int]:
        response = self.json("/tokenize", {"prompt": prompt, "add_special_tokens": False})
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
            raise RuntimeError(f"unexpected /tokenize response: {response!r}")
        return tokens

    def metrics_text(self) -> str | None:
        try:
            with self._request("/metrics") as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
