"""Pytest configuration — makes the modules in src/ importable from tests."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
