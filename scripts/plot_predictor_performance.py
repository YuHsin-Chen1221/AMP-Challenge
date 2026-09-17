#!/usr/bin/env python3
"""Challenge-report figure: predictor performance across versions v1-v3, single panel.
X-axis = the challenge's category-level endpoints on the HONEST (homology) split:
the four MIC categories (strain-count-weighted per-species Spearman rho via the panel
aggregation) plus the HC50 safety endpoint. Bars coloured by version.
Conventions: no title (caption carries it), series (version) by colour not marker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "challenge/metrics"))
from aggregate import category_rho  # noqa: E402

D = "/tmp/claude-1007/-home-yuhsin-Desktop/e7fadbea-08c5-43c8-9ea1-d0ca4743439c/scratchpad"
FILES = {"v1": "predictor_separate_homology.json",
         "v2": "predictor_separate_homology_v2.json",
         "v3": "predictor_separate_homology_v3.json"}

# category label -> key in category_rho output; HC50 handled separately
CATS = [("Broad", "broad"), ("Gram+", "gram_pos"), ("Gram$-$", "gram_neg"), ("MDR", "mdr")]
COL = {"v1": "#9ecae1", "v2": "#2c5aa0", "v3": "#8452b5"}   # colour = version

# assemble per-version category rho (MIC) + HC50 overall rho, homology split
data = {}
for v, fn in FILES.items():
    r = json.load(open(f"{D}/{fn}"))["results"]["homology"]
    psp = {s: m["spearman"] for s, m in r["mic"]["final"]["per_group"].items()}
    crho = category_rho(psp)
    row = [crho[key] for _, key in CATS]
    row.append(r["hemo"]["final"]["overall"]["spearman"])   # HC50
    data[v] = row

labels = [c[0] for c in CATS] + ["HC50"]
vers = ["v1", "v2", "v3"]

fig, ax = plt.subplots(figsize=(7.6, 3.8))
x = np.arange(len(labels)); w = 0.26
for i, v in enumerate(vers):
    b = ax.bar(x + (i - 1) * w, data[v], w, color=COL[v], label=v)
    for bar in b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.8)

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.72)
ax.set_ylabel("Spearman $\\rho$ (honest / homology split)")
ax.set_xlabel("challenge category (strain-weighted) $\\cdot$ safety endpoint")
ax.axhline(0, color="#888", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.legend(title="version", frameon=False, fontsize=9, title_fontsize=9, loc="upper right")
fig.tight_layout()

out = ROOT / "challenge/report/figures/predictor_performance"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
print("wrote", out)
for v in vers:
    print(v, {l: round(x, 3) for l, x in zip(labels, data[v])})
