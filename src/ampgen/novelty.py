"""Novelty checks against the official reference set (data/antibacterial.fasta), matching the
challenge validator (scripts/verify_submission.py):

  - full 50k library : NO sequence identical to any reference (exact-match filter)
  - top-100          : NO sequence with Levenshtein.ratio > 0.80 to any reference

Levenshtein.ratio is the official similarity metric (NOT alignment identity / CD-HIT). For the
library we only need exact-dedup (a set lookup); the >0.80 ratio gate is applied to top candidates.
"""
from __future__ import annotations

from pathlib import Path

import Levenshtein


def read_fasta(path: str | Path) -> list[str]:
    seqs, cur = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def exact_reference_set(fasta: str | Path) -> set[str]:
    """Reference sequences as a set, for the library's exact-match novelty filter."""
    return set(read_fasta(fasta))


def max_ratio(seq: str, references: list[str]) -> float:
    """Highest Levenshtein.ratio of `seq` against any reference (the quantity the top-100 gate
    thresholds at 0.80). O(N_ref) per query."""
    seq = seq.upper()
    best = 0.0
    for ref in references:
        r = Levenshtein.ratio(seq, ref)
        if r > best:
            best = r
            if best >= 1.0:
                break
    return best


def passes_top_gate(seq: str, references: list[str], threshold: float = 0.80) -> bool:
    return max_ratio(seq, references) <= threshold
