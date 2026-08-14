"""wristview: egocentric head-mounted video in, robot wrist-camera views out.

Import side effect, deliberate and load-bearing: torch and pycolmap each ship
their own copy of libomp. Loading both in one process aborts with OMP Error #15.
Set the escape hatch before any other module imports torch.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep OpenMP from oversubscribing the performance cores during COLMAP calls.
os.environ.setdefault("OMP_NUM_THREADS", "8")

__version__ = "0.1.0"
