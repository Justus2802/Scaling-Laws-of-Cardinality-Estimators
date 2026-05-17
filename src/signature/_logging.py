"""Package-level logger for the signature package.

Usage in block modules::

    from ._logging import get_logger
    log = get_logger(__name__)

Consumers configure verbosity via standard :mod:`logging`::

    import logging
    logging.getLogger("signature").setLevel(logging.DEBUG)
    logging.basicConfig()               # or attach any handler you like

Log levels used across the package:

* **INFO**  — one line per computed signature/sub-feature inside ``calculate()``,
  with headline result values (see ``block_b.py`` as the orientation example).
* **WARNING** — plot-failure fallback inside ``_visualize_plot()`` and other
  genuine fallback paths.
* **ERROR** — unexpected failures that fall back to NaN / default values.
"""

import logging


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger scoped to *module_name* under the 'signature' root.

    Args:
        module_name: typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` whose name is derived from *module_name*,
        parented under the ``signature`` root logger.
    """
    # Strip the package prefix so names stay concise, e.g. "signature.block_f"
    parts = module_name.split(".")
    try:
        idx = parts.index("signature")
        tail = ".".join(parts[idx:])
    except ValueError:
        tail = module_name
    return logging.getLogger(tail)
