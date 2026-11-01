"""Loose-intent tier tests (issue #23, no model, no data).

Pins the tier contract: 10 underdetermined items (no digits, no units),
1-2 compiling references each, a 3-check rubric each, and harness
helpers that keep the precise tier byte-compatible (legacy row ids,
ref hashes, runs CSV) while scoring intent Jaccard as max-over-refs.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import rq4  # noqa: E402
from eval.rq4_intent_items import INTENT_ITEMS  # noqa: E402
from eval.rq4_items import ITEMS  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

# Loose English must not smuggle the spec back in: no counts, windows,
# thresholds, amounts or time units.
BANNED = ("USD", "%", "hour", "day", "hop", "transfer", "window",
          "threshold", "span", "emit", "within", "trapezoid", "ramp")


def test_intent_tier_shape() -> None:
    assert len(INTENT_ITEMS) == 10
    ids = [it["id"] for it in INTENT_ITEMS]
    assert len(set(ids)) == 10 and all(i.startswith("intent-") for i in ids)
    precise_ids = {it["id"] for it in ITEMS}
    assert not (set(ids) & precise_ids)  # ref-cache row ids can't collide
    for it in INTENT_ITEMS:
        assert 1 <= len(it["references"]) <= 2
        assert len(it["rubric"]) == 3
        for check in it["rubric"]:
            assert check["id"] and check["criterion"]


def test_intent_english_underdetermines_the_motif() -> None:
    for it in INTENT_ITEMS:
        english = it["english"]
        assert not re.search(r"\d", english), it["id"]  # no counts at all
        lowered = english.lower()
        for token in BANNED:
            assert token.lower() not in lowered, (it["id"], token)


def test_intent_references_compile() -> None:
    for it in INTENT_ITEMS:
        for ref in it["references"]:
            doc = parse_text(ref)
            assert [m.name for m in doc.motifs] == ["ref"], it["id"]
            compile_motif(doc, "ref")


def test_precise_tier_stays_legacy_compatible() -> None:
    wrapped = rq4.tier_items("precise")
    assert len(wrapped) == 50
    assert wrapped[0]["references"] == [ITEMS[0]["reference"]]
    # Ref hash is still sha256(single reference): old runs rows resume.
    assert rq4.tier_ref_hash(wrapped[0]["references"]) == hashlib.sha256(
        ITEMS[0]["reference"].encode()).hexdigest()[:16]
    assert rq4.tier_runs_csv("precise") == rq4.RUNS_CSV
    assert rq4.tier_runs_csv("intent") == rq4.INTENT_RUNS_CSV
    assert rq4.INTENT_RUNS_CSV != rq4.RUNS_CSV  # never merged
    assert rq4.tier_compiled_json("intent") != rq4.tier_compiled_json("precise")


def test_motif_sidecar_roundtrip(tmp_path) -> None:
    path = tmp_path / "runs_motifs.json"
    rq4._save_motif_text(path, "a", "repair", True, "motif cand {}")
    rq4._save_motif_text(path, "a", "oneshot", False, None)
    rq4._save_motif_text(path, "a", "repair", True, "motif cand { v2 }")
    rows = rq4._motif_sidecar(path)
    assert len(rows) == 2  # same (item, arm) overwrites
    assert [r for r in rows if r["arm"] == "repair"][0]["motif"].endswith("v2 }")
    rq4._drop_motif_texts(path, ["repair"])
    assert [r["arm"] for r in rq4._motif_sidecar(path)] == ["oneshot"]
    rq4._drop_motif_texts(path, ["oneshot"])
    assert not path.exists()


def test_max_jaccard_over_disjunction() -> None:
    assert rq4._max_jaccard([{1, 2}, {2, 3}], {2, 3}) == 1.0
    assert rq4._max_jaccard([{1}, {2}], {3}) == 0.0
    assert rq4._max_jaccard([set(), set()], {1}) is None  # no ref fires
    assert rq4._max_jaccard([set(), {1}], {1}) == 1.0  # empty ref skipped
    assert rq4._max_jaccard([{1, 2}], set()) == 0.0  # uncompiled candidate
