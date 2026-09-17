"""Programmatic implementation of the AMP Challenge 2027 peptide design constraints.

This is the single source of truth for what makes a sequence a *legal* candidate.
Passing these is NECESSARY but NOT SUFFICIENT for being a good candidate
(ranking by an activity/selectivity oracle happens downstream).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
LEN_MIN = 8
LEN_MAX = 50

# Metadata tokens (any casing) that mark a peptide as violating the topology /
# modification rules. Used when a curated DB exposes modification/synthesis fields.
_MOD_PATTERNS = re.compile(
    r"cyclic|cyclis|cyclot|disulf|amidat|acetylat|lipid|myrist|palmit|"
    r"glycos|pegylat|staple|dendrim|d-amino|d-form|beta-|non-?standard|"
    r"unusual|modified",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str = ""


def clean_sequence(seq: str) -> str:
    """Uppercase, strip whitespace/gaps/stops. Does not alter residues."""
    if seq is None:
        return ""
    return re.sub(r"[\s\-\*\.]", "", str(seq)).upper()


def check_sequence(seq: str) -> CheckResult:
    """Check a bare sequence against alphabet + length + topology-detectable-from-sequence.

    Modifications that are NOT encodable in a plain 20-letter string
    (staple, lipid, PEG, terminal amidation) cannot be seen here — they must be
    filtered from curated-DB metadata via `metadata_flags_modification`.
    """
    s = clean_sequence(seq)
    if not s:
        return CheckResult(False, "empty")
    n = len(s)
    if n < LEN_MIN:
        return CheckResult(False, f"too_short({n})")
    if n > LEN_MAX:
        return CheckResult(False, f"too_long({n})")
    bad = set(s) - STANDARD_AA
    if bad:
        return CheckResult(False, f"non_canonical_aa({''.join(sorted(bad))})")
    return CheckResult(True, "ok")


def metadata_flags_modification(*fields: object) -> bool:
    """True if any provided metadata field mentions a disallowed modification/topology."""
    for f in fields:
        if f is None:
            continue
        if _MOD_PATTERNS.search(str(f)):
            return True
    return False


def is_valid(seq: str) -> bool:
    return check_sequence(seq).ok


# ---- DataFrame helpers -------------------------------------------------------

def filter_dataframe(df, seq_col="sequence", meta_cols=None, dedup=True):
    """Return (kept_df, rejection_summary_dict).

    Adds a cleaned `sequence` and drops rows that fail constraints. If `meta_cols`
    are given, rows whose metadata flags a modification are also dropped.
    """
    import pandas as pd  # local import: keep module importable without pandas

    meta_cols = meta_cols or []
    df = df.copy()
    df[seq_col] = df[seq_col].map(clean_sequence)

    results = df[seq_col].map(check_sequence)
    df["_ok"] = results.map(lambda r: r.ok)
    df["_reason"] = results.map(lambda r: r.reason)

    if meta_cols:
        present = [c for c in meta_cols if c in df.columns]
        if present:
            modflag = df[present].apply(
                lambda row: metadata_flags_modification(*row.values), axis=1
            )
            df.loc[modflag & df["_ok"], "_reason"] = "modified_metadata"
            df.loc[modflag, "_ok"] = False

    reasons = df.loc[~df["_ok"], "_reason"].value_counts().to_dict()
    kept = df[df["_ok"]].drop(columns=["_ok", "_reason"]).reset_index(drop=True)

    n_before = len(kept)
    if dedup:
        kept = kept.drop_duplicates(subset=[seq_col]).reset_index(drop=True)
    summary = {
        "input": int(len(df)),
        "rejected": {k: int(v) for k, v in reasons.items()},
        "kept_pre_dedup": int(n_before),
        "duplicates_removed": int(n_before - len(kept)),
        "kept": int(len(kept)),
    }
    return kept, summary
