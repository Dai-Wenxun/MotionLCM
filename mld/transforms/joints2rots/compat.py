"""Compatibility for the legacy Chumpy objects in the supplied SMPL pickle."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def legacy_numpy_aliases():
    """Expose removed NumPy aliases only while loading the old SMPL model."""
    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }
    missing = {name: value for name, value in aliases.items() if name not in np.__dict__}
    try:
        for name, value in missing.items():
            setattr(np, name, value)
        yield
    finally:
        for name in missing:
            delattr(np, name)
