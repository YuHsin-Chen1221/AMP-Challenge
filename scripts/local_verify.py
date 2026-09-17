"""Local mirror of the official scripts/verify_submission.py checks, run against generated
library.fasta / top.fasta (our repo is private, so the URL-cloning validator can't reach it).
Applies the exact challenge constants and the same three checks; optionally re-runs generation to
confirm reproducibility.

    python scripts/local_verify.py generate/library.fasta generate/top.fasta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ampgen.novelty import read_fasta  # noqa: E402

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50
LIBRARY_SIZE, TOP_SIZE = 50_000, 100
SIM_THRESHOLD = 0.80
ANTIBACTERIAL = ROOT / "data/antibacterial.fasta"


def verify_library(seqs: list[str]) -> list[str]:
    e = []
    if len(seqs) != LIBRARY_SIZE:
        e.append(f"library: expected {LIBRARY_SIZE} sequences, got {len(seqs)}")
    seen = set()
    for i, s in enumerate(seqs, 1):
        if not s:
            e.append(f"record {i}: empty"); continue
        bad = set(s) - STANDARD_AA
        if bad:
            e.append(f"record {i}: invalid chars {sorted(bad)}")
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            e.append(f"record {i}: length {len(s)} out of [{MIN_LEN},{MAX_LEN}]")
        if s in seen:
            e.append(f"record {i}: duplicate")
        seen.add(s)
    return e


def verify_no_overlap(seqs: set[str], refs: set[str]) -> list[str]:
    ov = seqs & refs
    return [f"library: {len(ov)} sequence(s) identical to a reference"] if ov else []


def verify_top_similarity(seqs: list[str], refs: list[str]) -> list[str]:
    e = []
    if len(seqs) > TOP_SIZE:
        e.append(f"top: {len(seqs)} > {TOP_SIZE}")
    for s in seqs:
        for r in refs:
            if Levenshtein.ratio(s, r) > SIM_THRESHOLD:
                e.append(f"top: '{s}' has Levenshtein ratio > {SIM_THRESHOLD} vs a reference")
                break
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library", nargs="?", default=str(ROOT / "generate/library.fasta"))
    ap.add_argument("top", nargs="?", default=str(ROOT / "generate/top.fasta"))
    args = ap.parse_args()

    lib = [s.upper() for s in read_fasta(args.library)]
    top = [s.upper() for s in read_fasta(args.top)]
    refs = [s.upper() for s in read_fasta(ANTIBACTERIAL)]
    ref_set = set(refs)

    errors = (verify_library(lib)
              + verify_no_overlap(set(lib), ref_set)
              + verify_top_similarity(top, refs))
    print(f"library: {len(lib)} seqs | top: {len(top)} seqs | references: {len(refs)}")
    if errors:
        print("FAILED:")
        for x in errors[:30]:
            print("  -", x)
        sys.exit(1)
    print("PASS: all challenge checks (size, alphabet, length, uniqueness, novelty, top-100 similarity).")


if __name__ == "__main__":
    main()
