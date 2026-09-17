"""Physicochemical descriptors for AMP distribution analysis.

Used to (a) characterise the mining pool vs curated AMPs, and later
(b) compare against generative-model output distributions. Encodes the
biophysical "essence" levers: charge, hydrophobicity, and amphipathicity
(Eisenberg hydrophobic moment) plus its sequence-periodicity form.
"""
from __future__ import annotations

import math

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Eisenberg consensus hydrophobicity (used for the hydrophobic moment)
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}
POS = frozenset("KR")   # ignore His (mostly neutral at pH 7.4)
NEG = frozenset("DE")
HYDROPHOBIC = frozenset("AILMFWVC")
AROMATIC = frozenset("FWY")


def net_charge(seq: str) -> int:
    return sum(1 for a in seq if a in POS) - sum(1 for a in seq if a in NEG)


def gravy(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD.get(a, 0.0) for a in seq) / len(seq)


def frac_hydrophobic(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in HYDROPHOBIC) / len(seq)


def frac_aromatic(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in AROMATIC) / len(seq)


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Eisenberg mean hydrophobic moment <uH> at a given periodicity.

    100 deg = alpha-helix. This is the magnitude of the Fourier component of the
    hydrophobicity signal at the helical frequency — the amphipathicity lever.
    Returned per-residue-normalised so lengths are comparable.
    """
    if not seq:
        return 0.0
    ang = math.radians(angle_deg)
    sx = sum(EISENBERG.get(a, 0.0) * math.cos(i * ang) for i, a in enumerate(seq))
    sy = sum(EISENBERG.get(a, 0.0) * math.sin(i * ang) for i, a in enumerate(seq))
    return math.hypot(sx, sy) / len(seq)


def max_hydrophobic_run(seq: str) -> int:
    """Longest contiguous hydrophobic stretch — tracks hemolysis risk."""
    best = cur = 0
    for a in seq:
        cur = cur + 1 if a in HYDROPHOBIC else 0
        best = max(best, cur)
    return best


def descriptors(seq: str) -> dict:
    """Full descriptor row for one sequence."""
    n = len(seq)
    return {
        "length": n,
        "net_charge": net_charge(seq),
        "charge_per_res": net_charge(seq) / n if n else 0.0,
        "gravy": gravy(seq),
        "frac_hydrophobic": frac_hydrophobic(seq),
        "frac_aromatic": frac_aromatic(seq),
        "uH_alpha": hydrophobic_moment(seq, 100.0),   # amphipathicity (helix)
        "uH_beta": hydrophobic_moment(seq, 160.0),    # amphipathicity (sheet)
        "max_hydrophobic_run": max_hydrophobic_run(seq),
    }


def descriptor_frame(seqs):
    import pandas as pd
    return pd.DataFrame(descriptors(s) for s in seqs)
