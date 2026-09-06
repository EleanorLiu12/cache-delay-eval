"""Small dependency-free client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionTiming:
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


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

    def version(self) -> str | None:
        try:
            response = self.json("/version")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
        if isinstance(response, dict):
            value = response.get("version")
            return str(value) if value is not None else None
        return None

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

    def stream_completion(
        self,
        model: str,
        prompt: str | list[int],
        max_tokens: int,
        seed: int,
    ) -> CompletionTiming:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start_ns = time.perf_counter_ns()
        first_token_ns: int | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        with self._request("/v1/completions", payload) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                event = json.loads(line)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    text = choice.get("text")
                    if text and first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        end_ns = time.perf_counter_ns()
        if first_token_ns is None:
            raise RuntimeError("stream ended without a non-empty token")
        return CompletionTiming(
            ttft_ms=(first_token_ns - start_ns) / 1_000_000,
            e2e_ms=(end_ns - start_ns) / 1_000_000,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )


def parse_prometheus(text: str | None) -> dict[str, float]:
    """Sum Prometheus samples by metric name, ignoring labels."""
    values: dict[str, float] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def metric(metrics: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None
