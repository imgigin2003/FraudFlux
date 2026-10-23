"""Vercel entry point: serves the FraudFlux API as a Python function.

vercel.json rewrites /api/* here; FastAPI routes on the original path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.main import app  # noqa: E402,F401
