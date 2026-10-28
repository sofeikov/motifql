"""Muse Spark client over the Meta Model API (Oct 31, compiler step 1).

OpenAI-compatible chat completions at https://api.meta.ai/v1 with the key
from `MUSE_KEY`. Every call is cached on disk by a hash of the canonical
request (DESIGN.md §10: cache every model call by prompt hash), so reruns
and harness retries never repay. stdlib only (urllib).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("spark")

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
CACHE_DIR = Path("eval/results/cache/models")


class SparkError(RuntimeError):
    """The Model API call failed (after retries) or the key is missing."""


def _key() -> str:
    key = os.environ.get("MUSE_KEY", "").strip()
    if not key:
        raise SparkError("MUSE_KEY is not set; export it before calling the model.")
    return key


def _canonical(model: str, messages: list[dict], max_tokens: int,
               reasoning_effort: str | None) -> str:
    return json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens,
                       "reasoning_effort": reasoning_effort},
                      sort_keys=True, separators=(",", ":"))


def cache_path(request_json: str, cache_dir: Path = CACHE_DIR) -> Path:
    digest = hashlib.sha256(request_json.encode()).hexdigest()[:32]
    return cache_dir / f"{digest}.json"


def complete(messages: list[dict], *, model: str = DEFAULT_MODEL,
             max_tokens: int = 4000,
             reasoning_effort: str | None = "low",
             cache_dir: Path = CACHE_DIR,
             timeout_s: float = 180.0,
             attempts: int = 3) -> tuple[str, dict]:
    """One chat completion; returns (text, usage dict).

    Cache hits return the stored text with its original usage plus
    `"cached": True`. Reasoning models burn completion tokens before any
    content, so keep max_tokens generous.
    """
    req = _canonical(model, messages, max_tokens, reasoning_effort)
    path = cache_path(req, cache_dir)
    if path.exists():
        record = json.loads(path.read_text())
        usage = dict(record.get("usage", {}))
        usage["cached"] = True
        return record["text"], usage
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {_key()}",
               "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            http_req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=data, headers=headers,
                method="POST")
            with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode())
            text = payload["choices"][0]["message"]["content"] or ""
            usage = dict(payload.get("usage", {}))
            usage["cached"] = False
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request": json.loads(req), "text": text,
                                        "usage": usage}))
            return text, usage
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            last = SparkError(f"HTTP {exc.code}: {detail}")
            flaky_filter = exc.code == 400 and "content_policy_violation" in detail
            if exc.code in (429,) or 500 <= exc.code < 600 or flaky_filter:
                # The content filter misfires on benign AML-detection
                # prompts (identical prompt fails then passes), so 400s
                # of that one kind are retried like 429s.
                time.sleep(2.0 * (attempt + 1))
                continue
            raise last from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = SparkError(f"transport error: {exc}")
            time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]
