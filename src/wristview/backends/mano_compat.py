"""Make chumpy importable on Python 3.11 and NumPy 1.26.

WiLoR predicts MANO parameters, and running the MANO layer needs the
`MANO_RIGHT.pkl` model. That pickle holds chumpy arrays, so unpickling it
imports chumpy. chumpy was last released in 2019 and breaks twice on a modern
stack:

  1. it imports `numpy.bool`, `numpy.int`, `numpy.float`, and friends, all
     removed in NumPy 1.24
  2. it calls `inspect.getargspec`, removed in Python 3.11

Both are restorable aliases, not real incompatibilities. `apply()` puts them
back before chumpy is imported. Import this module before anything that
touches smplx or WiLoR.
"""

from __future__ import annotations

import warnings

_APPLIED = False

# The NumPy scalar aliases chumpy expects, mapped to their builtin equivalents.
_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}


def apply() -> None:
    """Install the shims. Safe to call more than once."""
    global _APPLIED
    if _APPLIED:
        return

    import inspect

    import numpy as np

    # NumPy warns loudly that these names are reserved for future use. That
    # warning is aimed at new code, and this is a 2019 dependency we cannot
    # edit, so silence it rather than print it once per alias per process.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        for name, alias in _NUMPY_ALIASES.items():
            if not hasattr(np, name):
                setattr(np, name, alias)

    # Removed in Python 3.11. getfullargspec is the superset that replaced it,
    # and chumpy only reads `.args`, which both provide.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    _APPLIED = True


def available() -> tuple[bool, str]:
    """Report whether chumpy and smplx can be imported after the shims."""
    try:
        apply()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import chumpy  # noqa: F401
            import smplx  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any failure means unavailable
        return False, f"{type(exc).__name__}: {exc}"
    return True, "available"
