"""AMPGen submission entry point: `uv run generate`.

Produces, reproducibly (fixed seed), two FASTA files matching the challenge validator:
  generate/library.fasta  -- 50,000 unique, constraint-valid, novel (no exact reference match)
  generate/top.fasta      -- top-100 ranked, each <=0.80 Levenshtein ratio to any reference

Pipeline: activity-BLIND generation (masked-LM Gibbs sampler, length from the generator's own
distribution, hard 8-50 gate) -> structural/dedup/exact-novelty filter -> 50k library -> score
with the dual-endpoint predictor (10-species MIC + HC50) -> panel aggregation -> single composite
(broad-spectrum Success Rate + Safety Window, safety-gated) -> Levenshtein novelty gate -> top-100.
Generation and the predictor are kept separate on purpose (the library is robust to the scorer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import predict as predict_mod
from .aggregate import aggregate_panel
from .constraints import check_sequence, clean_sequence
from .novelty import exact_reference_set, read_fasta, passes_top_gate
from .ranking import rank_indices

ROOT = Path(__file__).resolve().parents[2]
ENCODER = ROOT / "weights/generator"
ANTIBACTERIAL = ROOT / "data/antibacterial.fasta"
LENGTH_DIST = ROOT / "data/length_dist.json"
OUT_DIR = ROOT / "generate"


# --------------------------------------------------------------------------- #
# generation: masked-LM Gibbs sampler (activity-blind, generator's own prior)
# --------------------------------------------------------------------------- #
def _aa_token_ids(tok):
    return [tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"]


def sample_lengths(n: int, rng: np.random.Generator) -> np.ndarray:
    d = json.loads(Path(LENGTH_DIST).read_text())
    lens = np.array([int(k) for k in d], dtype=int)
    p = np.array([d[str(k)] for k in lens], dtype=float); p /= p.sum()
    return rng.choice(lens, size=n, p=p)


@torch.no_grad()
def gibbs_generate(model, tok, lengths, steps, temperature, remask_frac, rng, device, batch=256):
    """De-novo Gibbs: start all-<mask> at length L, iteratively predict+sample-fill (temperature),
    remasking a fraction each step. Sampling is restricted to the 20 AA tokens."""
    aa_ids = torch.tensor(_aa_token_ids(tok), device=device)
    cls, eos, mask = tok.cls_token_id, tok.eos_token_id, tok.mask_token_id
    seqs = []
    order = np.argsort(lengths)                         # group equal lengths into batches
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = int(lengths[idx][0]) if len(set(lengths[idx].tolist())) == 1 else None
        # split further by exact length so a batch is rectangular
        for Lval in sorted(set(lengths[idx].tolist())):
            sub = idx[lengths[idx] == Lval]
            B = len(sub)
            ids = torch.full((B, Lval + 2), mask, device=device)
            ids[:, 0] = cls; ids[:, -1] = eos
            for step in range(steps):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
                lg = logits[:, 1:-1, :]                  # residue positions
                sel = torch.full_like(lg, float("-inf"))
                sel[..., aa_ids] = lg[..., aa_ids]       # restrict to 20 AAs
                probs = torch.softmax(sel / temperature, dim=-1)
                samp = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(B, Lval)
                masked = ids[:, 1:-1] == mask
                ids[:, 1:-1][masked] = samp[masked]
                if step < steps - 1:                     # remask a fraction for the next pass
                    rm = torch.rand(B, Lval, device=device) < remask_frac
                    ids[:, 1:-1][rm] = mask
            for b in range(B):
                seqs.append(tok.decode(ids[b, 1:-1]).replace(" ", "").upper())
    return seqs


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def valid_novel(seqs, reference_set, seen):
    out = []
    for s in seqs:
        s = clean_sequence(s)
        if not check_sequence(s).ok:                     # 20 AA, 8-50, linear
            continue
        if s in seen or s in reference_set:              # unique + no exact reference match
            continue
        seen.add(s); out.append(s)
    return out


def write_fasta(path, seqs, prefix):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{s}\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=50_000)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length", type=int, default=50)            # hard max gate (challenge cap)
    ap.add_argument("--oversample", type=float, default=1.6)     # generate extra to net n after filtering
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--remask-frac", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ENCODER))
    gen = AutoModelForMaskedLM.from_pretrained(str(ENCODER)).to(device).eval()

    reference_set = exact_reference_set(ANTIBACTERIAL)
    references = read_fasta(ANTIBACTERIAL)

    # 1. generate + filter until we have n unique valid novel sequences
    library, seen = [], set()
    while len(library) < args.n_sequences:
        need = args.n_sequences - len(library)
        lengths = sample_lengths(int(need * args.oversample) + 64, rng)
        lengths = np.clip(lengths, 8, args.length)
        cand = gibbs_generate(gen, tok, lengths, args.steps, args.temperature,
                              args.remask_frac, rng, device)
        library += valid_novel(cand, reference_set, seen)
        print(f"[generate] library {min(len(library), args.n_sequences)}/{args.n_sequences}", flush=True)
    library = library[:args.n_sequences]
    write_fasta(OUT_DIR / "library.fasta", library, "ampgen")
    print(f"[generate] wrote {OUT_DIR/'library.fasta'}", flush=True)

    # 2. score every library member -> panel aggregation -> composite rank
    rows = predict_mod.score(library, device=device)
    panels = [r["panel"] for r in rows]
    ranked = rank_indices(panels)                        # best-first by composite (safety-gated)

    # 3. walk the ranking, keep those passing the Levenshtein novelty gate, until top-k
    top = []
    for i in ranked:
        s = library[i]
        if passes_top_gate(s, references, threshold=0.80):
            top.append(s)
            if len(top) >= args.top_k:
                break
    write_fasta(OUT_DIR / "top.fasta", top, "ampgen_top")
    print(f"[generate] wrote {OUT_DIR/'top.fasta'} ({len(top)} sequences)", flush=True)


if __name__ == "__main__":
    main()
