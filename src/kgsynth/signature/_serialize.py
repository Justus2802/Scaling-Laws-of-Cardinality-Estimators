"""JSON (de)serialization helpers for signature blocks.

Each block stores rich internal state — numpy arrays, ``PowerLawStats`` tuples,
nested dicts keyed by ``int`` or ``str``, and tuples — none of which survive a
plain ``json.dump``/``json.load`` round-trip. The helpers here encode that state
into JSON-compatible structures (tagged dicts) and decode them back to the exact
original types, so a computed block can be persisted and rebuilt without
recomputation.
"""

import numpy as np

from ._utils import PowerLawStats

# Marker key identifying a tagged, non-trivial encoded value.
_TYPE_KEY = "__sig_type__"


def encode(obj: object) -> object:
    """Recursively convert a value into a JSON-serializable structure.

    Handles the concrete types found in block state; plain JSON scalars, lists,
    and ``None`` pass through unchanged. Raises ``TypeError`` for anything else
    so unexpected state is caught rather than silently dropped.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, np.ndarray):
        return {_TYPE_KEY: "ndarray", "dtype": str(obj.dtype), "data": obj.tolist()}
    if isinstance(obj, PowerLawStats):
        return {_TYPE_KEY: "PowerLawStats", "data": [float(x) for x in obj]}
    if isinstance(obj, tuple):
        return {_TYPE_KEY: "tuple", "data": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        # Encoded as an items list so non-string (e.g. int) keys survive.
        return {_TYPE_KEY: "dict", "items": [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def decode(obj: object) -> object:
    """Inverse of :func:`encode` — rebuild original types from tagged structures."""
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if isinstance(obj, dict):
        tag = obj.get(_TYPE_KEY)
        if tag is None:
            return {k: decode(v) for k, v in obj.items()}
        if tag == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"])
        if tag == "PowerLawStats":
            return PowerLawStats(*obj["data"])
        if tag == "tuple":
            return tuple(decode(x) for x in obj["data"])
        if tag == "dict":
            return {decode(k): decode(v) for k, v in obj["items"]}
        raise ValueError(f"Unknown serialized type tag {tag!r}")
    return obj


def encode_state(state: dict) -> dict:
    """Encode a block's ``__dict__`` (str-keyed) into a JSON object."""
    return {key: encode(value) for key, value in state.items()}


def decode_state(data: dict) -> dict:
    """Decode an :func:`encode_state` payload back into a block ``__dict__``."""
    return {key: decode(value) for key, value in data.items()}
