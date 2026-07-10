"""Package-level logger for the motif_counter package.

Consumers configure verbosity via standard :mod:`logging`::

    import logging
    logging.getLogger("motif_counter").setLevel(logging.INFO)

Log levels used across the package:

* **INFO**    — counting progress and graph size summaries.
* **WARNING** — fallback paths (e.g. ESCAPE falling back to CC sampling due
  to high-degree hub nodes).
"""

import logging


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger scoped to *module_name* under the 'motif_counter' root."""
    parts = module_name.split(".")
    try:
        idx = parts.index("motif_counter")
        tail = ".".join(parts[idx:])
    except ValueError:
        tail = module_name
    return logging.getLogger(tail)
