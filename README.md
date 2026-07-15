# kgsynth

Synthetic knowledge-graph generator that matches a real KG's statistical signature: measure a
real KG, generate a synthetic one that targets the same signature, and compare the two.

## Install

```bash
pip install -e .
```

Requires Python ≥3.10. See `requirements.txt` / `pyproject.toml` for dependencies.

## Quickstart

Three core operations, as an installed CLI:

```bash
kgsynth measure  data/graphs/swdf/swdf.nt
kgsynth generate swdf --seed 42 --rewire-budget 50000
kgsynth compare  graph_a.ttl graph_b.ttl
```

Or as minimal runnable Python scripts — see [`examples/`](examples/):

```bash
python examples/measure_a_kg.py
python examples/generate_and_compare.py
```

## Documentation

- [`user_docs/`](user_docs/) — using `kgsynth` as a library or CLI: the signature, the generator,
  the dataset/perturbation pipeline, the API reference. Start here.
- [`developer_docs/`](developer_docs/) — contributing to the package, and the empirical
  investigations and design decisions behind it. Start here if you're extending `kgsynth` or
  researching why it works the way it does.
- [`scripts/`](scripts/) — research tooling beyond the CLI: parameter sweeps, convergence logging,
  diagnostic plots.

## Tests

```bash
pytest
```
