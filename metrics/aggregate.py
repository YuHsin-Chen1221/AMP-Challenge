"""Standalone shim -> the canonical implementation in src/ampgen/aggregate.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ampgen.aggregate import *          # noqa: F401,F403
from ampgen.aggregate import aggregate_panel, from_scores, category_rho, CATEGORY_WEIGHTS  # noqa: F401
if __name__ == "__main__":
    import ampgen.aggregate as a
    a.__dict__.get("__name__")
