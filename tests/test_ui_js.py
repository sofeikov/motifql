"""UI helper tests: pure `ui/app.js` functions exercised through node.

The demo page keeps its shared logic in node-harnessable pure
helpers (`module.exports`); this test runs them in-process via the
system node instead of duplicating the logic in Python. Skipped
where node is unavailable (the manual harness in the Nov 7 demo-ui
report covers that case).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
NODE = shutil.which("node")


def _run(script: str) -> object:
    out = subprocess.run(
        [NODE, "-e", script], capture_output=True, text=True, timeout=60,
        cwd=UI_DIR)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_union_witnesses_merges_time_ordered_with_shared() -> None:
    script = """
const { unionWitnesses } = require('./app.js');
const h = (tid, ts) => ({transfer_id: tid, timestamp: ts});
const { hops, shared } = unionWitnesses([
  [h(3, 30), h(1, 10)],
  [h(2, 20), h(3, 30)],
]);
console.log(JSON.stringify({
  order: hops.map((x) => x.transfer_id),
  shared: [...shared].sort((a, b) => a - b),
}));
"""
    assert _run(script) == {"order": [1, 2, 3], "shared": [3]}


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_union_witnesses_empty_and_disjoint() -> None:
    script = """
const { unionWitnesses } = require('./app.js');
const h = (tid, ts) => ({transfer_id: tid, timestamp: ts});
const empty = unionWitnesses([[], []]);
const apart = unionWitnesses([[h(1, 5)], [h(2, 6)]]);
console.log(JSON.stringify({
  empty: [empty.hops.length, empty.shared.size],
  apart: [apart.hops.map((x) => x.transfer_id), apart.shared.size],
}));
"""
    assert _run(script) == {"empty": [0, 0], "apart": [[1, 2], 0]}


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_union_witnesses_ignores_intra_list_repeats() -> None:
    # Group witnesses repeat a transfer across roles (leg1 + leg2);
    # that must not mark it shared — only presence in 2+ shapes does.
    script = """
const { unionWitnesses } = require('./app.js');
const h = (tid, ts) => ({transfer_id: tid, timestamp: ts});
const { hops, shared } = unionWitnesses(
  [[h(1, 5), h(1, 5)], [h(2, 6)]]);
console.log(JSON.stringify({
  order: hops.map((x) => x.transfer_id),
  shared: [...shared],
}));
"""
    assert _run(script) == {"order": [1, 2], "shared": []}
