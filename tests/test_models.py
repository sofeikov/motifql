"""Tests for the Spark client and repair loop (Oct 31, no network).

Transport is stubbed or cache-fed everywhere: nothing here may call the
Model API. The live path is exercised by eval/rq4.py, not the suite.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval.rq4_items import ITEMS  # noqa: E402
from motifql.models import repair, spark  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

GOOD = """term many = ramp(3, 8)
motif cand { match (s:account) -[e:transfer]-> (r:account)
 group by s over r window 1d
 having count(distinct r) >= 2
 soft count(distinct r) is many
 score product emit when score >= 0.5 }"""

TIMES_UP = """motif cand { match (a:account) -[e:transfer*2..5]-> (a)
 where e is time_ordered and span(e) <= 7d
 soft each e: e.gap is soon
 score min emit when score >= 0.6 }"""


def test_all_rq4_references_compile() -> None:
    # The English set's references are the RQ4 oracle: every one must
    # compile standalone (no model, no data).
    assert len(ITEMS) == 50
    for it in ITEMS:
        compile_motif(parse_text(it["reference"]), "ref")


def test_system_prompt_carries_grammar_and_examples() -> None:
    prompt = repair.system_prompt()
    assert "match" in prompt and "transfer" in prompt  # grammar present
    for name in ("round_trip", "fan_out", "scatter_gather", "layering_chain"):
        assert f"motif {name}" in prompt
    assert "term soon" in prompt


def test_try_compile_accepts_valid_and_names_errors() -> None:
    compiled, errors = repair.try_compile(GOOD)
    assert compiled is not None and errors == []
    assert compiled.plan is not None and compiled.libs is not None
    # Wrong motif name.
    _, errors = repair.try_compile(GOOD.replace("motif cand", "motif other"))
    assert len(errors) == 1 and "cand" in errors[0]
    # Garbage is a parse error, not an exception.
    _, errors = repair.try_compile("motif cand { match ((")
    assert len(errors) == 1 and errors[0].startswith("parse error:")
    # Missing time bound surfaces the E_TIME check through E_CHECK.
    _, errors = repair.try_compile(
        "motif cand { match (a:account) -[e:transfer*2..5]-> (a) }")
    assert len(errors) == 1 and "E_TIME" in errors[0]


def test_repair_loop_succeeds_first_try() -> None:
    calls = []

    def fake_complete(messages):
        calls.append(messages)
        return GOOD, {"total_tokens": 10}

    res = repair.repair_loop("a fan", fake_complete, max_attempts=3)
    assert res.compiled is not None and res.attempts == 1
    assert res.errors == [] and res.usage == [{"total_tokens": 10}]


def test_repair_loop_feeds_errors_back() -> None:
    seen = []

    def fake_complete(messages):
        seen.append(messages[-1]["content"])
        if len(seen) == 1:
            return "motif cand { match ((", {"total_tokens": 1}
        return GOOD, {"total_tokens": 2}

    res = repair.repair_loop("a fan", fake_complete, max_attempts=3)
    assert res.compiled is not None and res.attempts == 2
    assert "parse error" in seen[1]  # second prompt carries the feedback


def test_repair_loop_reports_unfixable() -> None:
    res = repair.repair_loop("a fan", lambda m: ("nope {", {}), max_attempts=2)
    assert res.compiled is None and res.attempts == 2
    assert len(res.errors) == 1


def test_sanity_crash_becomes_feedback_not_exception() -> None:
    def bad_sanity(compiled):
        raise ValueError("boom")

    res = repair.repair_loop("a fan", lambda m: (GOOD, {}),
                             max_attempts=2, sanity=bad_sanity)
    assert res.compiled is not None and res.attempts == 2
    assert "scoring crashed" in res.errors[0] and "boom" in res.errors[0]


def test_spark_client_caches_and_reports_usage(tmp_path, monkeypatch) -> None:
    payload = {"choices": [{"message": {"content": "motif cand {}"}}],
               "usage": {"total_tokens": 42}}
    hits = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            hits.append(1)
            return json.dumps(payload).encode()

    monkeypatch.setenv("MUSE_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResp())
    messages = [{"role": "user", "content": "hi"}]
    text, usage = spark.complete(messages, cache_dir=tmp_path)
    assert text == "motif cand {}" and usage["cached"] is False
    text2, usage2 = spark.complete(messages, cache_dir=tmp_path)
    assert text2 == text and usage2["cached"] is True
    assert len(hits) == 1  # second call served from disk
    assert usage2["total_tokens"] == 42


def test_spark_client_retries_flaky_content_filter(tmp_path, monkeypatch) -> None:
    # The provider content filter misfires on benign prompts (identical
    # prompt fails then passes), so that one 400 kind is retried.
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    calls = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 400, "bad", {}, io.BytesIO(b'{"code":"content_policy_violation"}'))
        return FakeResp()

    monkeypatch.setenv("MUSE_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    text, _ = spark.complete([{"role": "user", "content": "hi"}], cache_dir=tmp_path)
    assert text == "ok" and len(calls) == 2


def test_spark_client_requires_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MUSE_KEY", raising=False)
    try:
        spark.complete([{"role": "user", "content": "hi"}], cache_dir=tmp_path)
    except spark.SparkError as exc:
        assert "MUSE_KEY" in str(exc)
    else:
        raise AssertionError("expected SparkError without a key")


def test_spark_client_retries_then_raises(tmp_path, monkeypatch) -> None:
    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, io.BytesIO(b"busy"))

    monkeypatch.setenv("MUSE_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda s: None)
    try:
        spark.complete([{"role": "user", "content": "hi"}], cache_dir=tmp_path,
                       attempts=2)
    except spark.SparkError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("expected SparkError after retries")
