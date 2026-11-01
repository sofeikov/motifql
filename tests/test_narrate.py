"""Witness-narrative tests (issue #23, no network).

The model transport is stubbed or cache-fed everywhere: prompt
construction from the witness table, weakest-step extraction, the
fail-closed paths (missing data, model errors, declined or
hallucinated replies), and disk caching through the Spark client.
"""

from __future__ import annotations

import functools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.models import narrate, spark  # noqa: E402

ENGLISH = "Money leaves an account and comes back to it."
MOTIF = "motif cand { match (a:account) -[e:transfer*2..5]-> (a) }"


def _hop(tid: int, epoch: int, frm: str, to: str, amount: float) -> dict:
    return {"transfer_id": tid, "timestamp": epoch, "from_bank": "001",
            "from_account": frm, "to_bank": "001", "to_account": to,
            "amount": amount}


def _witness() -> dict:
    base = int(datetime(2022, 9, 3, 10, tzinfo=timezone.utc).timestamp())
    return {
        "hops": [_hop(11, base, "A", "B", 1000.0),
                 _hop(12, base + 3600, "B", "C", 990.0),
                 _hop(13, base + 9000, "C", "A", 980.0)],
        "explanation": [
            {"predicate": "each e: e.gap is patient", "degree": 0.841,
             "step": None},
            {"predicate": "each e: e.amount ~= prev.amount within 20%",
             "degree": 0.718, "step": 3},
        ],
        "top_score": 0.718,
        "found": True,
    }


FAITHFUL = ("$1,000 left 001/A for 001/B on 2022-09-03, then $990 moved to "
            "001/C an hour later and $980 returned to 001/A, closing the loop "
            "within 2.5h. The gap condition held at 0.841 and the amount "
            "condition at 0.718. The weakest step is hop 3, where the amount "
            "dipped furthest from the previous hop.")


def test_prompt_carries_exact_witness_figures() -> None:
    messages = narrate.build_messages(ENGLISH, MOTIF, _witness())
    assert messages[0]["role"] == "system"
    assert "only source of facts" in messages[0]["content"].lower() or \
        "only source" in messages[0]["content"]
    user = messages[1]["content"]
    for figure in ("$1,000", "$990", "$980", "2022-09-03", "001/A",
                   "001/B", "001/C", "0.841", "0.718", "hop 3"):
        assert figure in user, figure
    assert "CANNOT NARRATE" in messages[0]["content"]


def _group_witness() -> dict:
    base = int(datetime(2022, 9, 1, 6, tzinfo=timezone.utc).timestamp())
    return {
        "hops": [_hop(21, base, "S", "M1", 500.0),
                 _hop(22, base + 3600, "S", "M2", 510.0),
                 _hop(23, base + 7200, "M1", "D", 495.0)],
        "explanation": [
            {"predicate": "count(distinct m) is several", "degree": 1.0,
             "step": None},
        ],
        "top_score": 1.0,
        "found": True,
    }


def test_group_table_marks_shape_and_skips_gaps() -> None:
    assert narrate.is_chain(_witness()["hops"]) is True
    assert narrate.is_chain(_group_witness()["hops"]) is False
    table = narrate.witness_table(_group_witness(), ENGLISH, MOTIF)
    assert "SHAPE: group" in table
    assert "MEMBERS (3)" in table
    assert "gap after hop" not in table  # gaps are meaningless off-path
    assert "WEAKEST CONDITION" in table  # no hop number to name
    # Distinct counts, so the model never hand-counts repeated accounts.
    assert "DISTINCT: 2 senders, 3 recipients across 3 transfers" in table
    assert "DISTINCT" not in narrate.witness_table(_witness(), ENGLISH, MOTIF)
    chain_table = narrate.witness_table(_witness(), ENGLISH, MOTIF)
    assert "SHAPE: path" in chain_table and "HOPS (3)" in chain_table
    assert "gap after hop 1" in chain_table
    assert "WEAKEST STEP" in chain_table


def test_group_narrative_verifies() -> None:
    text = ("001/S sent $500 to 001/M1 and $510 to 001/M2 on 2022-09-01, "
            "and 001/M1 forwarded $495 to 001/D. The count condition held "
            "at 1.000. The weakest condition is the member count at 1.000.")
    res = narrate.narrate(ENGLISH, MOTIF, _group_witness(),
                          complete=lambda m: (text, {}))
    assert res.text == text and res.issues == []
    assert res.weakest_step is None  # groups often carry no hop number


def test_happy_path_keeps_text_and_weakest_step() -> None:
    seen = []

    def stub(messages):
        seen.append(messages)
        return FAITHFUL, {"total_tokens": 5}

    res = narrate.narrate(ENGLISH, MOTIF, _witness(), complete=stub)
    assert res.text == FAITHFUL and res.issues == []
    assert (res.weakest_step, res.weakest_degree) == (3, 0.718)
    assert "amount" in (res.weakest_predicate or "")
    assert res.usage == {"total_tokens": 5}
    assert len(seen) == 1


def test_fail_closed_on_missing_data() -> None:
    calls = []

    def stub(messages):
        calls.append(1)
        raise AssertionError("model must not be called")

    empty = dict(_witness(), hops=[])
    assert narrate.narrate(ENGLISH, MOTIF, empty, complete=stub).text is None
    broken = _witness()
    del broken["hops"][1]["amount"]
    res = narrate.narrate(ENGLISH, MOTIF, broken, complete=stub)
    assert res.text is None and any("amount" in i for i in res.issues)
    assert narrate.narrate("", MOTIF, _witness(), complete=stub).text is None
    assert narrate.narrate(ENGLISH, "  ", _witness(), complete=stub).text is None
    assert narrate.narrate(ENGLISH, MOTIF, {"hops": "x"},
                           complete=stub).text is None
    assert calls == []


def test_fail_closed_on_model_errors_and_declines() -> None:
    def down(messages):
        raise spark.SparkError("refused")

    res = narrate.narrate(ENGLISH, MOTIF, _witness(), complete=down)
    assert res.text is None and "unavailable" in res.issues[0]

    def decline(messages):
        return "CANNOT NARRATE", {}

    assert narrate.narrate(ENGLISH, MOTIF, _witness(),
                           complete=decline).text is None

    def empty(messages):
        return "   ", {}

    assert narrate.narrate(ENGLISH, MOTIF, _witness(),
                           complete=empty).text is None


def test_hallucinated_amounts_and_dates_are_dropped() -> None:
    richer = FAITHFUL.replace("$980", "$9,800")
    res = narrate.narrate(ENGLISH, MOTIF, _witness(),
                          complete=lambda m: (richer, {}))
    assert res.text is None
    assert any("$9,800" in i for i in res.issues)

    shifted = FAITHFUL.replace("2022-09-03", "2022-09-04")
    res = narrate.narrate(ENGLISH, MOTIF, _witness(),
                          complete=lambda m: (shifted, {}))
    assert res.text is None
    assert any("2022-09-04" in i for i in res.issues)

    # Non-strict keeps the text with its issues: spot-check input only.
    res = narrate.narrate(ENGLISH, MOTIF, _witness(),
                          complete=lambda m: (richer, {}), strict=False)
    assert res.text == richer and len(res.issues) == 1


def test_weakest_step_extraction() -> None:
    assert narrate.weakest([]) is None
    rows = [{"predicate": "a", "degree": 0.9, "step": None},
            {"predicate": "b", "degree": 0.5, "step": 2}]
    assert narrate.weakest(rows) == ("b", 0.5, 2)
    tied = [{"predicate": "a", "degree": 0.5, "step": 1},
            {"predicate": "b", "degree": 0.5, "step": 2}]
    assert narrate.weakest(tied)[0] == "a"  # ties go to the first


VERDICT = ("This is a round trip scoring 0.718: money leaves an account "
             "and comes back through 3 hops with similar amounts. The "
             "weakest link is the amount similarity at 0.718.")


def test_verdict_prompt_bans_code_names() -> None:
    messages = narrate.build_verdict_messages(ENGLISH, MOTIF, _witness())
    system = messages[0]["content"]
    assert "predicate code names" in system
    assert "DistinctArg" in system  # named as a forbidden example
    assert "CANNOT NARRATE" in system
    user = messages[1]["content"]
    assert "one-line verdict" in user
    for figure in ("HOPS (3)", "0.841", "0.718"):
        assert figure in user, figure


def test_summarize_happy_path_and_fail_closed() -> None:
    res = narrate.summarize(ENGLISH, MOTIF, _witness(),
                            complete=lambda m: (VERDICT, {"cached": True}))
    assert res.text == VERDICT and res.issues == []
    assert res.usage.get("cached") is True
    # Same fail-closed contract as narrate().

    def stub(messages):
        raise AssertionError("model must not be called")

    empty = dict(_witness(), hops=[])
    assert narrate.summarize(ENGLISH, MOTIF, empty, complete=stub).text is None
    richer = VERDICT + " The total was $9,800."
    res = narrate.summarize(ENGLISH, MOTIF, _witness(),
                            complete=lambda m: (richer, {}))
    assert res.text is None and any("$9,800" in i for i in res.issues)


DIAGNOSIS = ("A round trip scoring 0.718 overlaps a group scoring 1.0: "
              "the $990 hop also fans out among $510 transfers, so one "
              "criminal structure fires both shapes. The weakest link is "
              "the amount similarity at 0.718.")


def test_diagnosis_prompt_stacks_shapes_with_shared() -> None:
    w1, w2 = _witness(), _group_witness()
    messages = narrate.build_diagnosis_messages(
        [ENGLISH, "a fan"], [MOTIF, MOTIF], [w1, w2])
    system = messages[0]["content"]
    assert "ONE criminal structure" in system
    assert "CANNOT NARRATE" in system
    user = messages[1]["content"]
    assert "SHAPE 1:" in user and "SHAPE 2:" in user
    assert "SHARED:" in user
    assert "$990" in user and "$510" in user  # both tables present
    with pytest.raises(ValueError):
        narrate.build_diagnosis_messages([ENGLISH], [MOTIF], [w1])


def test_summarize_composite_verifies_union() -> None:
    w1, w2 = _witness(), _group_witness()
    res = narrate.summarize_composite(
        [ENGLISH, "a fan"], [MOTIF, MOTIF], [w1, w2],
        complete=lambda m: (DIAGNOSIS, {}))
    assert res.text == DIAGNOSIS and res.issues == []
    assert res.weakest_step is None  # hops are per-shape
    assert res.weakest_degree == 0.718  # min across shapes
    forged = DIAGNOSIS + " Also $9,800 moved."
    res = narrate.summarize_composite(
        [ENGLISH, "a fan"], [MOTIF, MOTIF], [w1, w2],
        complete=lambda m: (forged, {}))
    assert res.text is None and any("$9,800" in i for i in res.issues)


def test_narration_is_cached_on_disk(tmp_path, monkeypatch) -> None:
    hits = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            hits.append(1)
            return json.dumps(
                {"choices": [{"message": {"content": FAITHFUL}}],
                 "usage": {}}).encode()

    monkeypatch.setenv("MUSE_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResp())
    complete = functools.partial(spark.complete, cache_dir=tmp_path)
    first = narrate.narrate(ENGLISH, MOTIF, _witness(), complete=complete)
    second = narrate.narrate(ENGLISH, MOTIF, _witness(), complete=complete)
    assert first.text == second.text == FAITHFUL
    assert len(hits) == 1  # second narration served from disk
    assert second.usage.get("cached") is True
