"""Shared package-level logger for every ``kgsynth`` subpackage.

Usage in any module::

    from .._logging import get_logger  # from signature/generator/motif_counter
    log = get_logger(__name__)

Consumers configure verbosity via standard :mod:`logging`, keyed on the
subpackage name (``signature``, ``generator``, or ``motif_counter``)::

    import logging
    logging.getLogger("signature").setLevel(logging.DEBUG)
    # Prefix every line with a timestamp; or attach any handler you like
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Log levels used across these packages:

* **INFO**    — one line per computed signature/sub-feature or stage step,
  with headline result values (see ``block_b.py`` as the orientation example).
* **WARNING** — genuine fallback paths (e.g. plot-failure fallback inside
  ``_visualize_plot()``, ESCAPE falling back to CC sampling, a stage that
  can't run and returns its input unchanged).
* **ERROR**   — unexpected failures that fall back to NaN / default values.
"""

import logging

# Every kgsynth subpackage using get_logger() below this root.
_SUBPACKAGES = ("signature", "generator", "motif_counter")


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger scoped to *module_name* under its subpackage root.

    Parameters
    ----------
    module_name : str
        Typically ``__name__`` of the calling module, e.g.
        ``kgsynth.signature.block_f``.

    Returns
    -------
    logging.Logger
        A logger whose name is truncated to start at the first recognized
        subpackage component, e.g. ``signature.block_f``, so verbosity can be
        configured per-subpackage regardless of the ``kgsynth.`` prefix.
    """
    parts = module_name.split(".")
    for pkg in _SUBPACKAGES:
        if pkg in parts:
            tail = ".".join(parts[parts.index(pkg):])
            return logging.getLogger(tail)
    return logging.getLogger(module_name)
