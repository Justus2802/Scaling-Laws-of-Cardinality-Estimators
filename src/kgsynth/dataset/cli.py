"""``kgsynth dataset`` — generate a dataset of perturbed synthetic KGs.

Wired into the top-level CLI by :func:`add_parser`; see ``user_docs/dataset.md``.
"""

import argparse

from .config import DatasetConfig
from .runner import describe, run


def _cmd_dataset(args: argparse.Namespace) -> int:
    """Run (or describe) a dataset config.

    :param args: Parsed CLI arguments.
    :returns: Number of failed units — the process exit code.
    """
    config = DatasetConfig.from_yaml(args.config)  # raises on anything invalid
    if args.out_dir:
        config = type(config)(**{**vars(config), "out_dir": args.out_dir})
    if args.measure:
        config = type(config)(**{**vars(config), "measure": True})

    if args.dry_run:
        print(describe(config))
        return 0
    return run(config, workers=args.workers, force=args.force)


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register the ``dataset`` subcommand on the top-level parser.

    :param sub: The subparsers action returned by ``add_subparsers``.
    """
    d = sub.add_parser(
        "dataset",
        help="Generate many synthetic KGs by perturbing one measured signature",
        description="Perturb a measured signature and generate one synthetic KG per "
                    "perturbation, in parallel. See user_docs/dataset.md.",
    )
    d.add_argument("config", help="YAML dataset config (see examples/perturb_dataset.yaml)")
    d.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes (default: CPU count)")
    d.add_argument("--measure", action="store_true",
                   help="Re-measure each generated graph and record per-block distances "
                        "(roughly doubles the per-graph cost)")
    d.add_argument("--force", action="store_true",
                   help="Regenerate graphs that already exist (default: skip and resume)")
    d.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without generating anything")
    d.add_argument("--out-dir", default=None,
                   help="Override the config's out_dir")
    d.set_defaults(func=_cmd_dataset)
