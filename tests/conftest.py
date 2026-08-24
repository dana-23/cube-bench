"""Make the src-layout package importable when running from a checkout.

``pip install -e .`` is the normal path; this only covers running pytest without
installing first, and is a no-op once the package is on sys.path.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
