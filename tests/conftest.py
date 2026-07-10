"""Pytest configuration for the kgsynth suite.

``kgsynth`` is imported from the installed distribution (``pip install -e .``) rather
than through a ``sys.path`` hack, so nothing needs to be wired up for it here.

This file's presence anchors pytest's rootdir. Combined with the absence of an
``__init__.py`` in this directory, pytest's default ``prepend`` import mode puts
``tests/`` on ``sys.path``, which is what lets a few tests import their
same-directory oracle helper (e.g. ``from _block_e_library_oracle import load_graph``).
"""
