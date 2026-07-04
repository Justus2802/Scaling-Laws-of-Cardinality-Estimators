#!/usr/bin/env python3
"""Sample a novel reduced signature from the measured corpus (doc-Stage-1).

Loads the measured signatures under ``data/graphs/<name>/signature/signature.json``
and draws one novel 88-feature signature with the v0 ``UniformRangeSampler``
(each feature ~ uniform over its corpus range, widened by ±10 %). Writes the
``{"source", "features"}`` JSON to a file or stdout — the same shape as a measured
``signature.json``, so it is drop-in compatible with the existing readers.

See ``docs/plan/stage1_population_sampler.md`` for the design.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signature_sampler import UniformRangeSampler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed for reproducibility."
    )
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="Corpus directory (default: data/graphs/).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the sampled signature here (default: stdout).",
    )
    args = parser.parse_args()

    sampler = (
        UniformRangeSampler.load_corpus(args.corpus)
        if args.corpus is not None
        else UniformRangeSampler.load_corpus()
    )
    features = sampler.sample(seed=args.seed)
    payload = sampler.to_json(features, source=f"sampled:UniformRangeSampler:seed={args.seed}")

    if args.out is not None:
        sampler.write(args.out, features, source=payload["source"])
        print(f"Wrote sampled signature ({len(features)} features) to {args.out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
