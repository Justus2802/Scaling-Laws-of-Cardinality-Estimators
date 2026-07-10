"""Package-level logger for the generator package.

Mirrors ``signature._logging``: stage modules call ``get_logger(__name__)`` and
consumers configure verbosity via standard :mod:`logging`::

    import logging
    logging.getLogger("generator").setLevel(logging.INFO)
    # Prefix every line with a timestamp; or attach any handler you like
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Log levels used across the package:

* **INFO**  — one line per stage with headline parameters / results
  (schema summary, instantiation counts, refinement progress).
* **WARNING** — genuine fallback paths (e.g. a stage that can't run and
  returns its input unchanged).
"""

import logging


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger scoped to *module_name* under the 'generator' root.

    Args:
        module_name: typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` parented under the ``generator`` root logger,
        e.g. ``generator.stage1``.
    """
    parts = module_name.split(".")
    try:
        idx = parts.index("generator")
        tail = ".".join(parts[idx:])
    except ValueError:
        tail = module_name
    return logging.getLogger(tail)
