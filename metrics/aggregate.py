"""Panel aggregation: per-species predictions -> 20-strain panel -> 4 challenge category scores.

The predictor outputs one MIC per species (10 species) and one HC50. The challenge scores a
20-strain panel; each panel strain maps to exactly one species, so a species' predicted MIC is
broadcast to each strain it represents (PANEL_STRAIN_COUNTS). Category scores are then computed
over the strain-expanded panel:

  Broad-Spectrum  -> all 20 strains
  Gram-Positive   -> the 5 Gram+ strains
  Gram-Negative   -> the 15 Gram- strains
  MDR ESKAPE      -> the 8-strain resistant subset (MDR_SUBSET_COUNTS)
  Optimal Select. -> Safety Window = HC50 / MIC50(broad)

Success Rate = fraction of (strain-weighted) strains with MIC <= potency threshold (16 uM).
MIC50/MIC90 = strain-weighted median / 90th percentile MIC. Values are clipped at the challenge
ceilings (MIC 64 uM, HC50 128 uM) before aggregation.

Two entry points:
  aggregate_panel(mic_uM_by_species, hc50_uM) -> per-category {success_rate, mic50, mic90} (+ selectivity)
  category_rho(per_species_rho)               -> strain-weighted category-level ranking metric
                                                 (used for the report figure)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ampgen.predictor import (  # noqa: E402
    SPECIES, GRAM, PANEL_STRAIN_COUNTS, MDR_SUBSET_COUNTS,
)

POTENCY_UM = 16.0        # MIC <= this counts as a hit
MIC_CEILING_UM = 64.0
HC50_CEILING_UM = 128.0

# category -> {species: n_strains represented in that category}
GRAM_POS = [s for s in SPECIES if GRAM[s] == "pos"]
GRAM_NEG = [s for s in SPECIES if GRAM[s] == "neg"]
CATEGORY_WEIGHTS = {
    "broad":    dict(PANEL_STRAIN_COUNTS),                                  # all 20
    "gram_pos": {s: PANEL_STRAIN_COUNTS[s] for s in GRAM_POS},              # 5
    "gram_neg": {s: PANEL_STRAIN_COUNTS[s] for s in GRAM_NEG},              # 15
    "mdr":      dict(MDR_SUBSET_COUNTS),                                    # 8
}
assert sum(CATEGORY_WEIGHTS["broad"].values()) == 20
assert sum(CATEGORY_WEIGHTS["gram_pos"].values()) == 5
assert sum(CATEGORY_WEIGHTS["gram_neg"].values()) == 15
assert sum(CATEGORY_WEIGHTS["mdr"].values()) == 8


def _expand(value_by_species: dict, weights: dict) -> np.ndarray:
    """Broadcast a per-species value to one entry per panel strain it represents."""
    out = []
    for sp, n in weights.items():
        if sp in value_by_species and value_by_species[sp] is not None:
            out += [float(value_by_species[sp])] * int(n)
    return np.array(out, dtype=float)


def aggregate_panel(mic_uM_by_species: dict, hc50_uM: float | None = None) -> dict:
    """Per-species predicted MIC (uM) -> category Success Rate / MIC50 / MIC90, plus Safety Window.

    mic_uM_by_species : {species_key: predicted MIC in uM}. Missing species are dropped from the
                        strain expansion (their strains do not contribute).
    hc50_uM           : predicted human-RBC HC50 in uM (for the Optimal-Selectivity category).
    """
    out = {}
    for cat, w in CATEGORY_WEIGHTS.items():
        v = _expand(mic_uM_by_species, w)
        if len(v) == 0:
            out[cat] = None
            continue
        v = np.minimum(v, MIC_CEILING_UM)
        out[cat] = {
            "success_rate": float(np.mean(v <= POTENCY_UM)),   # weighted by strain count
            "mic50": float(np.median(v)),
            "mic90": float(np.percentile(v, 90)),
            "n_strains": int(len(v)),
        }
    if hc50_uM is not None and out.get("broad"):
        hc = min(float(hc50_uM), HC50_CEILING_UM)
        mic50 = out["broad"]["mic50"]
        out["selectivity"] = {"safety_window": hc / mic50 if mic50 > 0 else float("inf"),
                              "hc50_uM": hc, "mic50_broad_uM": mic50}
    return out


def from_scores(mic_score_by_species: dict, hemo_score: float | None = None) -> dict:
    """Convenience: the predictor emits s_mic = -log10(MIC uM) and s_hemo = log10(HC50 uM).
    Convert scores -> uM and aggregate."""
    mic_uM = {sp: 10.0 ** (-s) for sp, s in mic_score_by_species.items()}
    hc50 = 10.0 ** hemo_score if hemo_score is not None else None
    return aggregate_panel(mic_uM, hc50)


def category_rho(per_species_rho: dict) -> dict:
    """Strain-count-weighted mean of per-species Spearman rho within each category -- the
    category-level ranking quality used for the report figure. Species absent from
    `per_species_rho` are skipped (weights renormalised over those present)."""
    out = {}
    for cat, w in CATEGORY_WEIGHTS.items():
        pairs = [(per_species_rho[s], n) for s, n in w.items() if s in per_species_rho]
        if not pairs:
            out[cat] = None
            continue
        rhos, ws = zip(*pairs)
        out[cat] = float(np.average(rhos, weights=ws))
    return out


if __name__ == "__main__":
    # self-test: a peptide potent on Gram- but weak on Gram+, safe (high HC50)
    demo = {"e. coli": 2.0, "p. aeruginosa": 4.0, "k. pneumoniae": 8.0, "a. baumannii": 4.0,
            "s. enterica": 8.0, "e. cloacae": 12.0,                     # Gram- : mostly hits
            "s. aureus": 40.0, "b. subtilis": 32.0, "e. faecalis": 50.0, "e. faecium": 60.0}  # Gram+ : misses
    agg = aggregate_panel(demo, hc50_uM=200.0)
    for k, v in agg.items():
        print(f"{k:12s} {v}")
    print("\ncategory_rho demo:",
          category_rho({s: 0.6 for s in SPECIES}))
