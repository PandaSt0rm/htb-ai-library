"""Pytest configuration for htb_ai_library tests."""

import os
import sys

import matplotlib


matplotlib.use("Agg", force=True)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
