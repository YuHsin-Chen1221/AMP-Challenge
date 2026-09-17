#!/usr/bin/env python3
"""Read the four Stage-2 dual-endpoint reports and emit the Markdown result tables.

Inputs (in --dir): predictor_{separate,multitask}_{homology,random}.json
Emits to stdout three tables:
  A  config comparison  (separate vs multitask x homology vs random, overall MIC + HC50)
  B  MIC per-species    (best config, homology & random, 10 species + overall + baseline)
  C  HC50 overall       (both modes x both splits + baseline)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIGS = ["separate", "multitask"]
SPLITS = ["homology", "random"]
# display order for MIC species: by homology test volume (large -> small)
SP_ORDER = ["e. coli", "s. aureus", "p. aeruginosa", "b. subtilis", "s. enterica",
            "k. pneumoniae", "e. faecalis", "e. cloacae", "e. faecium", "a. baumannii"]


def load(d: Path) -> dict:
    r = {}
    for mode in CONFIGS:
        for sp in SPLITS:
            f = d / f"predictor_{mode}_{sp}.json"
            if f.exists():
                rep = json.loads(f.read_text())
                r[(mode, sp)] = rep["results"].get(sp, {})
    return r


def ov(rep, ep):  # overall block for endpoint
    return rep.get(ep, {}).get("final", {}).get("overall")


def pg(rep, ep):  # per_group dict for endpoint
    return rep.get(ep, {}).get("final", {}).get("per_group", {})


def base(rep, ep):
    return rep.get(ep, {}).get("baseline", {}).get("overall")


def cap(s):
    return {"homology": "Homology (honest)", "random": "Random (inflated)"}[s]


def italic_sp(s):  # 'e. coli' -> '*E. coli*'
    return "*" + s[0].upper() + s[1:] + "*"


def table_a(R) -> str:
    L = ["### Config comparison — separate vs multitask, both endpoints\n",
         "Overall Spearman ρ / Pearson r / RMSE (log₁₀ µM), test set. ρ = ranking, "
         "r = absolute-magnitude calibration; both matter (the challenge scores an absolute "
         "16 µM potency threshold and the HC50/MIC50 safety window). *base* = mean-predictor "
         "RMSE (ρ=r=0). **Bold** = better honest (homology) value per endpoint.\n",
         "| Endpoint | Mode | Split | ρ | r | RMSE | base RMSE | n |",
         "|---|---|---|---|---|---|---|---|"]
    # find best honest per endpoint for bolding
    best = {}
    for ep in ("mic", "hemo"):
        vals = {m: (ov(R.get((m, "homology"), {}), ep) or {}).get("spearman", -9) for m in CONFIGS}
        best[ep] = max(vals, key=vals.get)
    for ep, lab in (("mic", "MIC"), ("hemo", "HC50")):
        for m in CONFIGS:
            for sp in SPLITS:
                o = ov(R.get((m, sp), {}), ep)
                b = base(R.get((m, sp), {}), ep)
                if not o:
                    continue
                bold = "**" if (sp == "homology" and m == best[ep]) else ""
                L.append(f"| {lab} | {m} | {cap(sp)} | {bold}{o['spearman']:.3f}{bold} | "
                         f"{o['pearson']:.3f} | {o['rmse']:.3f} | {b['rmse']:.3f} | {o['n']} |")
    return "\n".join(L)


def table_b(R, mode) -> str:
    L = [f"### MIC per-species — {mode} model\n",
         "Spearman ρ / Pearson r / RMSE / MAE (log₁₀ µM) per panel species, then overall and the "
         "mean-predictor baseline. n = test measurements for that species.\n",
         "| Species | n | ρ (hom) | r (hom) | RMSE (hom) | ρ (rand) | r (rand) |",
         "|---|---|---|---|---|---|---|"]
    hom = R.get((mode, "homology"), {})
    ran = R.get((mode, "random"), {})
    ph = pg(hom, "mic")
    pr = pg(ran, "mic")
    for s in SP_ORDER:
        h = ph.get(s); r = pr.get(s)
        if not h:
            continue
        rr = (f"{r['spearman']:.3f} | {r['pearson']:.3f}" if r else "— | —")
        L.append(f"| {italic_sp(s)} | {h['n']} | {h['spearman']:.3f} | {h['pearson']:.3f} | "
                 f"{h['rmse']:.3f} | {rr} |")
    oh = ov(hom, "mic"); orr = ov(ran, "mic")
    L.append(f"| **overall** | {oh['n']} | **{oh['spearman']:.3f}** | **{oh['pearson']:.3f}** | "
             f"**{oh['rmse']:.3f}** | **{orr['spearman']:.3f}** | **{orr['pearson']:.3f}** |")
    bh = base(hom, "mic"); br = base(ran, "mic")
    L.append(f"| *mean-predictor* | {bh['n']} | 0.000 | 0.000 | {bh['rmse']:.3f} | 0.000 | 0.000 |")
    return "\n".join(L)


def table_c(R) -> str:
    L = ["### HC50 (hemolysis) — both modes × both splits\n",
         "Human-RBC HC50, single head (no species conditioning). Same metrics; n = test peptides.\n",
         "| Mode | Split | ρ | r | RMSE | MAE | base RMSE | n |",
         "|---|---|---|---|---|---|---|---|"]
    for m in CONFIGS:
        for sp in SPLITS:
            o = ov(R.get((m, sp), {}), "hemo"); b = base(R.get((m, sp), {}), "hemo")
            if not o:
                continue
            L.append(f"| {m} | {cap(sp)} | {o['spearman']:.3f} | {o['pearson']:.3f} | "
                     f"{o['rmse']:.3f} | {o['mae']:.3f} | {b['rmse']:.3f} | {o['n']} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--mic_mode", default="separate", help="which model's per-species MIC table to show")
    args = ap.parse_args()
    R = load(Path(args.dir))
    have = sorted(f"{m}/{s}" for (m, s) in R)
    print(f"<!-- reports loaded: {have} -->\n")
    print(table_a(R), "\n")
    print(table_b(R, args.mic_mode), "\n")
    print(table_c(R))


if __name__ == "__main__":
    main()
