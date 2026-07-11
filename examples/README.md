# Examples

Minimal, runnable illustrations of the `kgsynth` public API — the
measure → generate → compare loop from the project proposal (§5). Both run with
no arguments against a small graph bundled in the repo, and both accept a path
to your own `.ttl`/`.nt` KG.

Install the package first (from the repo root):

```bash
pip install -e .
```

### `measure_a_kg.py` — the *measure* step
Reduces a KG to its 124-feature signature and prints a per-block summary.

```bash
python examples/measure_a_kg.py                       # bundled sample graph
python examples/measure_a_kg.py path/to/your_kg.ttl
```

### `generate_and_compare.py` — the full round-trip
Measures a target signature, generates a synthetic KG that targets it, then
re-measures the synthetic graph and reports each block's mean relative feature
error.

```bash
python examples/generate_and_compare.py               # bundled sample graph
python examples/generate_and_compare.py path/to/kg.ttl 7   # custom graph + seed
```

For the fuller research workflow (convergence/swap logging, sweeps, diagnostic
plots) see `scripts/` and its README; for the CLI equivalents of these two
operations see `kgsynth measure` / `kgsynth generate` / `kgsynth compare`.
