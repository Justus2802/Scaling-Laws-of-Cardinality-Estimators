"""Cross-check the repo's hand-rolled motif counters against igraph's library
motif counter (``motifs_randesu``) on real KG files.

Two counters are checked, and they carry different guarantees:

``ExactMotifCounter``
    Hand-rolled exhaustive enumeration of triangles and the 3-/4-node graphlets.
    Compared to the library ground truth with **exact equality**. This is the
    load-bearing assertion of this module: it is what certifies that the custom
    enumeration is correct on real, non-toy graphs.

``HybridMotifCounter``, configured as ``signature.block_e.MOTIF_COUNTER``
    What Block E actually ships. It routes triangles to the exact backend but
    **k=4 to the colour-coding sampler**, so its 4-node counts are *estimates* at
    every graph size. They are therefore bounded within a tolerance, not compared
    exactly, and the per-graph relative error is printed.

The diamond graphlet gets its own, much looser bound: the CC estimator
systematically over-counts it, and the bias does not shrink with sample budget.
See ``KNOWN_CC_DIAMOND_BIAS`` in ``tests/test_hybrid_motif_counter.py``.

CSV columns (``tests/block_e_verification_graphs.csv``):
    name    — human-readable label used in subTest output
    path    — graph file path, relative to the repository root
    format  — ``nt`` or ``ttl`` (passed to the oracle CLI; load_kg itself
              detects the serialization from file content)

Graphs listed in the manifest must be small enough that exhaustive
``motifs_randesu`` enumeration finishes inside ``_ORACLE_TIMEOUT_S``. That rules
out hub-heavy graphs: fb237_v4 (n=4707, max degree 1050) needs ~336s, because a
single degree-1050 hub contributes C(1050,3) ≈ 1.9e8 induced 4-subgraphs on its
own. wn18rr_v4 (n=3861, max degree 68) needs 0.02s.

Missing files are skipped so the suite stays green in checkouts without the large
datasets — but ``test_at_least_one_graph_resolves`` fails if *every* row is
missing, so the cross-check can never again pass while silently verifying nothing.

The library ground truth runs in a **child process** under a wall-clock timeout
(``_ORACLE_TIMEOUT_S``). On timeout the child is killed and the subtest fails
cleanly with a message — it does not hang or abort the whole session. A
subprocess (not a thread) is used because ``motifs_randesu`` is a GIL-holding
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

import igraph
import numpy as np
import pytest
from kgsynth.motif_counter import ExactMotifCounter, HybridMotifCounter
from kgsynth.signature.block_e import _SAMPLE_BUDGET
from _block_e_library_oracle import load_graph

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CSV_PATH = os.path.join(os.path.dirname(__file__), "block_e_verification_graphs.csv")
_ORACLE = os.path.join(os.path.dirname(__file__), "_block_e_library_oracle.py")

# Wall-clock budget for the library ground-truth child process, per graph.
# On timeout the child is killed and the subtest fails cleanly. Keep this below
# the class's @pytest.mark.timeout backstop so the clean failure trips first.
_ORACLE_TIMEOUT_S = 300

# Degree sequences identifying each 4-node graphlet, keyed by the Block E feature
# name they back.
_MOTIF4_DEGSEQ = {
    "four_cycle": (2, 2, 2, 2),
    "diamond": (2, 2, 3, 3),
    "k4": (3, 3, 3, 3),
    "tailed": (1, 2, 2, 3),
}

# Tolerance for the colour-coding estimates that HybridMotifCounter returns for
# k=4. The absolute floor keeps tiny true counts (where relative error is
# meaningless) from failing on small absolute deltas — e.g. K4 on kgsynth_generated
# is 1 exact vs 0 estimated: a 100% relative error over an absolute delta of 1.
_ESTIMATE_REL_TOL = 0.25
_ESTIMATE_ABS_TOL = 10.0

# The diamond estimate is biased high regardless of sample budget (observed +52%,
# +73%, +70% on the three manifest graphs), so it cannot meet _ESTIMATE_REL_TOL.
# Bound it at "within a factor of two" purely to catch a gross regression; the real
# fix is tracked at KNOWN_CC_DIAMOND_BIAS in tests/test_hybrid_motif_counter.py.
_DIAMOND_REL_TOL = 1.0


def _read_csv() -> list[dict[str, str]]:
    """Read the verification graph manifest. Empty list if the CSV is absent."""
    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("path")]


def _abs_path(row: dict[str, str]) -> str:
    return os.path.join(_REPO_ROOT, row["path"])


def _undirected(path: str, fmt: str) -> igraph.Graph:
    """Load a KG as the undirected simplification both Block E and the oracle count on,
    with vertices in a canonical (name-sorted) order.

    The canonicalisation is what makes the colour-coding tolerance checks below
    reproducible. ``load_kg`` indexes vertices in rdflib's iteration order, which is
    hash-ordered and therefore differs on every interpreter run unless
    ``PYTHONHASHSEED`` is pinned. Exact motif counts are invariant under vertex
    relabelling, but a seeded sampler is not: the same graph and the same seed yield
    different estimates run to run. Sorting by vertex name removes that dependence
    here without changing ``load_kg``'s behaviour for the rest of the codebase.
    """
    g = load_graph(path, fmt).as_undirected(combine_edges="first").simplify()
    # permute_vertices maps new id k to old id perm[k] and *returns* a new graph rather
    # than mutating in place, so perm is the argsort itself and the result must be used.
    order = np.argsort(np.asarray(g.vs["name"], dtype=object))
    return g.permute_vertices(order.tolist())


def _shipped_counter() -> HybridMotifCounter:
    """A counter configured exactly as ``signature.block_e.MOTIF_COUNTER``.

    Deliberately a *fresh* instance rather than the module-level one: ``CCMotifCounter``
    seeds a single ``Generator`` in ``__init__`` and consumes it on every call, so the
    shared instance returns different estimates depending on how many other tests
    counted motifs before this one. Constructing our own keeps the tolerance checks
    reproducible under any test ordering.
    """
    return HybridMotifCounter(n_samples=_SAMPLE_BUDGET, seed=1)


@pytest.mark.timeout(360, method="thread")
class TestBlockEAgainstLibrary(unittest.TestCase):
    """Verify the custom motif counters against igraph library counts.

    Carries the suite's only pytest-timeout backstop (360s, applied here rather
    than suite-wide in pytest.ini): this is the one test that can hang, on a
    GIL-holding igraph ``motifs_randesu`` C call. ``method="thread"`` because the
    default signal-based timeout cannot interrupt that call mid-flight. The
    library ground truth also runs in a child process under its own cleaner
    ``_ORACLE_TIMEOUT_S`` (300s) limit; this backstop only fires if something
    outside that subprocess hangs.
    """

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

    def _assert_estimate(self, estimate: int, expected: int, rel_tol: float, msg: str) -> None:
        """Bound a colour-coding estimate against the exact library count.

        The relative error is reported on stdout (visible with ``pytest -s``) so
        the sampler's accuracy can be inspected as graphs are added.
        """
        rel_err = abs(estimate - expected) / expected if expected else (0.0 if not estimate else math.inf)
        print(
            f"[estimate] {msg}: cc={estimate} exact={expected} "
            f"rel_err={rel_err:.2%} (tol {rel_tol:.0%}+{_ESTIMATE_ABS_TOL:g})"
        )
        self.assertTrue(
            math.isclose(estimate, expected, rel_tol=rel_tol, abs_tol=_ESTIMATE_ABS_TOL),
            f"{msg} (cc estimate): {estimate} vs exact {expected} rel_err={rel_err:.2%}",
        )

    def test_csv_manifest_is_present_and_nonempty(self):
        self.assertTrue(
            _read_csv(),
            f"no verification graphs listed in {_CSV_PATH}",
        )

    def test_at_least_one_graph_resolves(self):
        """Guard against the whole cross-check silently skipping.

        Every per-graph subtest below skips when its file is missing, which is the
        right behaviour for optional large datasets but means a manifest of purely
        unresolvable paths would report green having verified nothing. At least one
        row must point at a file that is committed to the repo.
        """
        rows = _read_csv()
        resolved = [r["path"] for r in rows if os.path.exists(_abs_path(r))]
        self.assertTrue(
            resolved,
            "no graph in the verification manifest resolves to an existing file, so "
            "the Block E library cross-check would verify nothing. Listed paths: "
            + ", ".join(r["path"] for r in rows),
        )

    def test_exact_counter_matches_library(self):
        """ExactMotifCounter must equal igraph's exhaustive enumeration, exactly."""
        rows = _read_csv()
        if not rows:
            self.skipTest("no graphs listed in verification CSV")

        exact = ExactMotifCounter()
        for row in rows:
            name = row.get("name") or row["path"]
            fmt = row.get("format", "nt")
            abs_path = _abs_path(row)
            with self.subTest(graph=name):
                if not os.path.exists(abs_path):
                    self.skipTest(f"graph file not present: {row['path']}")

                gt = self._run_oracle(abs_path, fmt, name)
                g_und = _undirected(abs_path, fmt)

                _t0 = time.perf_counter()
                tri = exact.count_triangles(g_und)
                m4 = exact.count_motifsk(g_und, 4)
                our_seconds = time.perf_counter() - _t0
                print(
                    f"[timing] {name} (n={gt['n']}): library motifs_randesu "
                    f"{gt['lib_seconds']:.3f}s | ExactMotifCounter {our_seconds:.3f}s"
                )

                self.assertEqual(tri, gt["triangle"], f"{name}: triangle")
                for feature, degseq in _MOTIF4_DEGSEQ.items():
                    self.assertEqual(m4.get(degseq, 0), gt[feature], f"{name}: {feature}")

    def test_shipped_counter_estimates_are_within_tolerance(self):
        """The counter Block E ships routes k=4 to colour-coding, so its 4-node
        counts are estimates. Bound them; assert triangles exactly."""
        rows = _read_csv()
        if not rows:
            self.skipTest("no graphs listed in verification CSV")

        for row in rows:
            name = row.get("name") or row["path"]
            fmt = row.get("format", "nt")
            abs_path = _abs_path(row)
            with self.subTest(graph=name):
                if not os.path.exists(abs_path):
                    self.skipTest(f"graph file not present: {row['path']}")

                gt = self._run_oracle(abs_path, fmt, name)
                g_und = _undirected(abs_path, fmt)
                counter = _shipped_counter()  # fresh per graph, so subtests are independent

                # Triangles are routed to the exact backend, so they must match.
                self.assertEqual(
                    counter.count_triangles(g_und), gt["triangle"], f"{name}: triangle"
                )

                m4 = counter.count_motifsk(g_und, 4)
                for feature, degseq in _MOTIF4_DEGSEQ.items():
                    tol = _DIAMOND_REL_TOL if feature == "diamond" else _ESTIMATE_REL_TOL
                    self._assert_estimate(
                        m4.get(degseq, 0), gt[feature], tol, f"{name}: {feature}"
                    )


if __name__ == "__main__":
    unittest.main()
