"""Cross-check Block E's custom motif counters against igraph's library motif
counter (``motifs_randesu``) on real KG files.

Block E hand-rolls exact 3-/4-node motif counts (triangles, 4-cycles, diamonds,
K4, tailed triangles) and star counts for speed on sparse KGs. This module
verifies those custom results against ground truth produced by independent
igraph library methods, for every graph listed in
``tests/block_e_verification_graphs.csv``.

CSV columns:
    name    — human-readable label used in subTest output
    path    — graph file path, relative to the repository root
    format  — ``nt`` or ``ttl`` (passed to the oracle CLI; load_kg itself
              detects the serialization from file content)

Triangles (``list_triangles``) and star counts (degree formula) are exact at
any graph size and are always checked exactly. The 4-node motif counts
(4-cycle, diamond, K4, tailed triangle) are exact only below ``_LARGE_N``
undirected nodes; above it Block E switches to color-coding estimates, so those
four counts are compared against the exact library values within a relative
tolerance (see ``_ESTIMATE_REL_TOL``). Missing files are skipped, so the suite
stays green in checkouts that don't have the large datasets.

The library ground truth (exact ``motifs_randesu`` enumeration) can be expensive
on large/dense graphs, so it runs in a **child process** under a wall-clock
timeout (``_ORACLE_TIMEOUT_S``). On timeout the child is killed and the subtest
fails cleanly with a message — it does not hang or abort the whole session.
A subprocess (not a thread) is used because ``motifs_randesu`` is a GIL-holding
igraph C call that a same-process thread could neither interrupt nor de-schedule.

Run with ``-s`` to see the per-graph runtime and estimate-accuracy report::

    .venv/bin/python -m pytest tests/test_signature_block_e_vs_library.py -v -s
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # for the sibling oracle module
from signature import BlockE
from signature.block_e import _LARGE_N
from _block_e_library_oracle import load_graph

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CSV_PATH = os.path.join(os.path.dirname(__file__), "block_e_verification_graphs.csv")
_ORACLE = os.path.join(os.path.dirname(__file__), "_block_e_library_oracle.py")

# Wall-clock budget for the library ground-truth child process, per graph.
# On timeout the child is killed and the subtest fails cleanly. Keep this below
# the pytest-timeout backstop in pytest.ini so the clean failure trips first.
_ORACLE_TIMEOUT_S = 300

# Above _LARGE_N undirected nodes Block E switches its 4-node motif counts
# (4-cycle, diamond, K4, tailed triangle) from exact enumeration to color-coding
# *estimates*, which carry sampling error. Compare those against the exact
# library counts with a relative tolerance instead of exact equality. These are
# coarse bounds for the sampler — tighten them with empirical data if a large
# graph is added to the manifest. The absolute floor keeps tiny true counts
# (where relative error is meaningless) from failing on small absolute deltas.
_ESTIMATE_REL_TOL = 0.25
_ESTIMATE_ABS_TOL = 10.0


def _read_csv() -> list[dict[str, str]]:
    """Read the verification graph manifest. Empty list if the CSV is absent."""
    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("path")]


class TestBlockEAgainstLibrary(unittest.TestCase):
    """Verify Block E custom motif counts equal igraph library counts."""

    def _run_oracle(self, path: str, fmt: str, name: str) -> dict:
        """Run the library ground-truth oracle in a child process, killing it
        and failing the subtest cleanly if it exceeds ``_ORACLE_TIMEOUT_S``."""
        try:
            proc = subprocess.run(
                [sys.executable, _ORACLE, path, fmt],
                capture_output=True,
                text=True,
                timeout=_ORACLE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            self.fail(
                f"{name}: library oracle (motifs_randesu) exceeded "
                f"{_ORACLE_TIMEOUT_S}s and was terminated"
            )
        if proc.returncode != 0:
            self.fail(f"{name}: library oracle failed (rc={proc.returncode}):\n{proc.stderr}")
        return json.loads(proc.stdout)

    def _assert_motif(self, custom: int, expected: int, estimated: bool, msg: str) -> None:
        """Assert a 4-node motif count matches the library ground truth.

        Exact equality below _LARGE_N; within the color-coding tolerance above it.
        For estimated counts the relative error is reported on stdout (visible
        with ``pytest -s``) so the sampler's accuracy can be inspected.
        """
        if not estimated:
            self.assertEqual(custom, expected, msg)
            return
        rel_err = abs(custom - expected) / expected if expected else float("inf") if custom else 0.0
        print(
            f"[estimate] {msg}: custom={custom} exact={expected} "
            f"rel_err={rel_err:.2%} (tol {_ESTIMATE_REL_TOL:.0%}+{_ESTIMATE_ABS_TOL:g})"
        )
        self.assertTrue(
            math.isclose(custom, expected, rel_tol=_ESTIMATE_REL_TOL, abs_tol=_ESTIMATE_ABS_TOL),
            f"{msg} (estimate): {custom} vs exact {expected} rel_err={rel_err:.2%}",
        )

    def test_csv_manifest_is_present_and_nonempty(self):
        self.assertTrue(
            _read_csv(),
            f"no verification graphs listed in {_CSV_PATH}",
        )

    def test_motif_counts_match_library(self):
        rows = _read_csv()
        if not rows:
            self.skipTest("no graphs listed in verification CSV")

        for row in rows:
            name = row.get("name") or row["path"]
            fmt = row.get("format", "nt")
            abs_path = os.path.join(_REPO_ROOT, row["path"])
            with self.subTest(graph=name):
                if not os.path.exists(abs_path):
                    self.skipTest(f"graph file not present: {row['path']}")

                # Library ground truth, computed out-of-process under a timeout.
                gt = self._run_oracle(abs_path, fmt, name)
                n = gt["n"]
                # Above _LARGE_N the 4-node motif counts are color-coding
                # estimates, so they are checked with a tolerance below.
                estimated = n > _LARGE_N

                # Our implementation: time the full Block E computation.
                g = load_graph(abs_path, fmt)
                _t0 = time.perf_counter()
                e = BlockE().calculate(g)
                our_seconds = time.perf_counter() - _t0

                print(
                    f"[timing] {name} (n={n}): library motifs_randesu "
                    f"{gt['lib_seconds']:.3f}s | Block E calculate {our_seconds:.3f}s"
                )

                # Triangles use list_triangles on the full graph — exact at any
                # size, so always an exact comparison.
                self.assertEqual(e.triangle_count, gt["triangle"], f"{name}: triangle")

                # 4-node motifs: exact below _LARGE_N, color-coding estimates above.
                self._assert_motif(e.four_cycle_count, gt["four_cycle"], estimated, f"{name}: four_cycle")
                self._assert_motif(e.diamond_count, gt["diamond"], estimated, f"{name}: diamond")
                self._assert_motif(e.k4_count, gt["k4"], estimated, f"{name}: k4")
                self._assert_motif(e.tailed_triangle_count, gt["tailed"], estimated, f"{name}: tailed_triangle")

                # Star counts are computed by Block E with a vectorized float64
                # formula, so very large counts (k-stars on high-degree hubs can
                # exceed 2**53) lose exact-integer precision. Verify with a small
                # relative tolerance rather than exact equality.
                for k in range(2, 11):
                    custom = e.star_counts.get(k, 0)
                    expected = gt["stars"][str(k)]
                    if expected == 0:
                        self.assertEqual(custom, 0, f"{name}: star_count k={k}")
                    else:
                        self.assertTrue(
                            math.isclose(custom, expected, rel_tol=1e-9),
                            f"{name}: star_count k={k}: {custom} != {expected} "
                            f"(rel err {abs(custom - expected) / expected:.2e})",
                        )


if __name__ == "__main__":
    unittest.main()
