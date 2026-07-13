"""Run a dataset plan across a process pool.

One process per graph. Generation is CPU-bound single-threaded Python (Stage 3's
annealing loop), so processes — not threads — are what buys parallelism here.
"""

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .._logging import get_logger
from .config import DatasetConfig
from .plan import WorkUnit, build_units
from .worker import UnitResult, run_unit

log = get_logger(__name__)

# Each worker is single-threaded Python; letting numpy/BLAS spawn its own threads
# inside every one of them just oversubscribes the cores. Set before the pool is
# created so the spawned children inherit it.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _pin_blas_threads() -> None:
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")


def _pending(units: list[WorkUnit], force: bool) -> list[WorkUnit]:
    """Drop units already on disk, so an interrupted run resumes instead of restarting.

    A unit counts as done when its ``meta.json`` exists: that file is written last,
    so a half-written directory (killed mid-generation) is correctly treated as
    unfinished and redone.
    """
    if force:
        for unit in units:
            shutil.rmtree(unit.out_dir, ignore_errors=True)
        return list(units)
    return [u for u in units if not (u.out_dir / "meta.json").exists()]


def run(config: DatasetConfig, *, workers: int | None = None, force: bool = False) -> int:
    """Generate every graph the config asks for.

    :param config: The validated config.
    :param workers: Process count (default: ``os.cpu_count()``).
    :param force: Regenerate graphs that already exist instead of skipping them.
    :returns: Number of failed units — the process exit code.
    """
    units = build_units(config)
    todo = _pending(units, force)
    skipped = len(units) - len(todo)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count() or 1
    workers = min(workers, max(1, len(todo)))

    log.info(
        "dataset: %d units (%d to run, %d already done), %d workers → %s",
        len(units), len(todo), skipped, workers, config.out_dir,
    )
    if not todo:
        log.info("dataset: nothing to do (pass --force to regenerate)")
        return 0

    _pin_blas_threads()
    manifest = config.out_dir / "manifest.jsonl"
    results: list[UnitResult] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool, manifest.open("a") as log_file:
        futures = {pool.submit(run_unit, u): u for u in todo}
        for future in as_completed(futures):
            result = future.result()  # run_unit never raises; it returns failures
            results.append(result)
            log_file.write(json.dumps(result.as_json()) + "\n")
            log_file.flush()  # a killed run must leave a readable manifest
            done += 1
            status = "ok" if result.ok else "FAILED"
            log.info(
                "[%d/%d] %-28s %-6s V=%-7d E=%-8d %5.1fs%s",
                done, len(todo), result.label, status,
                result.num_entities, result.num_edges, result.elapsed,
                f"  ({result.error})" if not result.ok else "",
            )

    failed = [r for r in results if not r.ok]
    saturated = [r for r in results if r.ok and r.saturated]

    log.info("dataset: %d ok, %d failed → %s", len(results) - len(failed),
             len(failed), config.out_dir)
    if saturated:
        # Not a failure, but it silently biases a sensitivity sweep: the feature
        # reads as perturbed while the generator saw (nearly) the baseline value.
        log.warning(
            "dataset: %d unit(s) had a perturbation mostly absorbed by domain clamps — "
            "their 'no effect' readings are not trustworthy. See meta.json:clamp_report.",
            len(saturated),
        )
    for result in failed:
        log.error("  unit %d (%s): %s", result.index, result.label, result.error)
    return len(failed)


def describe(config: DatasetConfig) -> str:
    """Render the plan as a table without generating anything (``--dry-run``).

    :param config: The validated config.
    :returns: The human-readable plan.
    """
    units = build_units(config)
    lines = [
        f"base      : {config.base}",
        f"design    : {config.design}",
        f"out_dir   : {config.out_dir}",
        f"measure   : {config.measure}",
        f"seed      : {config.seed}",
        f"units     : {len(units)}",
        "",
        f"  {'#':>4}  {'label':<32} {'perturb':>12} {'generate':>12}  exists",
        "  " + "─" * 76,
    ]
    for unit in units:
        exists = "yes" if (unit.out_dir / "meta.json").exists() else "-"
        lines.append(
            f"  {unit.index:>4}  {unit.label:<32} {unit.perturb_seed:>12} "
            f"{unit.generate_seed:>12}  {exists}"
        )
    return "\n".join(lines)


def load_manifest(out_dir: Path) -> list[dict]:
    """Read a finished (or in-progress) run's ``manifest.jsonl``.

    :param out_dir: The dataset directory.
    :returns: One dict per completed unit, in completion order.
    """
    path = Path(out_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
