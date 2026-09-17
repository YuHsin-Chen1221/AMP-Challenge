#!/usr/bin/env python3
"""Challenge-report figure: FAIR data ablation on a single fixed split.
Both arms use the identical v3 CD-HIT-80% split (same held-out test); they differ only in
TRAINING data -- Arm A = v2 data (GRAMPA + ampbench master), Arm B = v3 data (+ AMPBench-MT).
X-axis = challenge category (strain-weighted per-species MIC Spearman rho via panel aggregation)
plus the HC50 safety endpoint; bars coloured by data arm. Removes the re-clustering confound, so
any difference is purely the added data. Conventions: no title, series by colour."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = "/tmp/claude-1007/-home-yuhsin-Desktop/e7fadbea-08c5-43c8-9ea1-d0ca4743439c/scratchpad"
F = json.load(open(f"{D}/fair_category.json"))   # {'v2data':{...}, 'v3data':{...}}

CATS = [("Broad", "broad"), ("Gram+", "gram_pos"), ("Gram$-$", "gram_neg"),
        ("MDR", "mdr"), ("HC50", "HC50")]
ARMS = [("v2 data\n(GRAMPA+master)", "v2data", "#9ecae1"),
        ("v3 data\n(+AMPBench-MT)", "v3data", "#8452b5")]

fig, ax = plt.subplots(figsize=(7.6, 3.9))
x = np.arange(len(CATS)); w = 0.38
for i, (lab, key, col) in enumerate(ARMS):
    vals = [F[key][c[1]] for c in CATS]
    b = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=lab)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
# delta annotations above each category
for j, (_, key1) in enumerate([(c[0], c[1]) for c in CATS]):
    d = F["v3data"][key1] - F["v2data"][key1]
    ax.text(x[j], max(F["v2data"][key1], F["v3data"][key1]) + 0.045,
            f"{d:+.3f}", ha="center", fontsize=7.5, fontweight="bold",
            color=("#2c7a2c" if d > 0.005 else "#888"))

ax.set_xticks(x); ax.set_xticklabels([c[0] for c in CATS])
ax.set_ylim(0, 0.62)
ax.set_ylabel("Spearman $\\rho$ (fixed v3 split — same test)")
ax.set_xlabel("challenge category (strain-weighted) $\\cdot$ safety endpoint")
ax.axhline(0, color="#888", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
fig.tight_layout()

out = ROOT / "challenge/report/figures/predictor_performance"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print("wrote", out)
