"""Top-100 ranking: a single, transparent composite over the separately-predicted endpoints.

Pipeline (per sequence): predict 10-species MIC + HC50 (separately, debuggable) -> panel
aggregation (Success Rate / MIC50 / Safety Window) -> ONE composite score -> rank. General
"broad-spectrum + selectivity" objective (no single category locked):

  potency = broad-spectrum Success Rate      (fraction of 20 panel strains with MIC <= 16 uM)
  safety  = log10 Safety Window = log10(HC50 / MIC50_broad)
  composite = z(potency) + z(safety)         (standardized across the library so the two axes
                                              combine on equal footing)

with a SAFETY GATE for top-100 eligibility: Safety Window >= `safety_floor` (default 1.0), so a
potent-but-hemolytic peptide (window < 1: lyses RBCs at a lower dose than it kills bacteria) can
never rank at the top. To instead target a single category, swap `potency` for that category's
Success Rate (gram_pos / gram_neg / mdr).
"""
from __future__ import annotations

import math

import numpy as np


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def composite_scores(panels: list[dict], potency_key: str = "broad",
                     safety_floor: float = 1.0, w_potency: float = 1.0,
                     w_safety: float = 1.0) -> np.ndarray:
    """panels = list of aggregate_panel(...) dicts (one per sequence). Returns a composite score
    array (higher = better). Sequences failing the safety gate get -inf so they sort last."""
    pot = np.array([(p.get(potency_key) or {}).get("success_rate", 0.0) for p in panels])
    sw = np.array([(p.get("selectivity") or {}).get("safety_window", 0.0) for p in panels])
    logsw = np.array([math.log10(max(v, 1e-3)) for v in sw])
    score = w_potency * _z(pot) + w_safety * _z(logsw)
    score = np.where(sw >= safety_floor, score, -np.inf)   # gate out toxic-but-potent
    return score


def rank_indices(panels: list[dict], **kw) -> np.ndarray:
    """Indices of sequences sorted best-first by the composite score."""
    return np.argsort(-composite_scores(panels, **kw))
