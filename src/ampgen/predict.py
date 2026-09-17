#!/usr/bin/env python3
"""Score peptides with the AMPGen dual-endpoint predictor (v2, separate mode).

Loads the frozen generator encoder (weights/generator/) and applies each saved head's
trainable delta (weights/predictor_mic.pt, weights/predictor_hemo.pt). Outputs, per sequence:
per-species MIC (uM), HC50 (uM), and the panel-aggregated category scores.

    python scripts/predict.py "KRWWKWWRR" "GLLDFLKKL" ...
    python scripts/predict.py --fasta library.fasta --out scores.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metrics"))
from .predictor import SpeciesConditionedScorer, SPECIES, SPECIES_TO_ID  # noqa: E402
from .aggregate import aggregate_panel                                          # noqa: E402

ENCODER = str(ROOT / "weights/predictor_encoder")
MIC_CKPT = ROOT / "weights/predictor_mic.pt"
HEMO_CKPT = ROOT / "weights/predictor_hemo.pt"


def _load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config", {})
    hd = tuple(int(x) for x in cfg["head_dims"].split(",")) if cfg.get("head_dims") else (None,)
    ll = tuple(int(x) for x in cfg["lora_layers"].split(",")) if cfg.get("lora_layers") else (8, 9, 10, 11)
    model = SpeciesConditionedScorer(ENCODER, pool=cfg.get("pool", "mean"), head_dims=hd, lora_layers=ll)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    # 'missing' = the frozen base-encoder keys (loaded from ENCODER already) -> expected.
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return model.to(device).eval()


@torch.no_grad()
def score(seqs, device="cpu"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    mic_model, hemo_model = _load(MIC_CKPT, device), _load(HEMO_CKPT, device)

    def enc(batch):
        e = tok(batch, return_tensors="pt", padding=True, return_special_tokens_mask=True)
        res = e["attention_mask"].clone(); res[e["special_tokens_mask"].bool()] = 0
        return e["input_ids"].to(device), e["attention_mask"].to(device), res.to(device)

    rows = []
    for i in range(0, len(seqs), 128):
        b = [s.upper() for s in seqs[i:i + 128]]
        ids, am, rm = enc(b)
        # MIC: score under each species' AdaLN -> per-species -log10(MIC)
        mic_scores = {}
        for sp in SPECIES:
            sid = torch.full((len(b),), SPECIES_TO_ID[sp], device=device)
            mic_scores[sp] = mic_model(ids, am, sid, rm).cpu()
        hemo = hemo_model.forward_hemo(ids, am, rm).cpu()   # log10(HC50)
        for j, s in enumerate(b):
            mic_uM = {sp: float(10.0 ** (-mic_scores[sp][j])) for sp in SPECIES}
            hc50_uM = float(10.0 ** hemo[j])
            agg = aggregate_panel(mic_uM, hc50_uM)
            rows.append({"sequence": s, "mic_uM": mic_uM, "hc50_uM": hc50_uM, "panel": agg})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*")
    ap.add_argument("--fasta")
    ap.add_argument("--out")
    args = ap.parse_args()
    seqs = list(args.seqs)
    if args.fasta:
        seqs += [l.strip() for l in open(args.fasta) if l.strip() and not l.startswith(">")]
    if not seqs:
        ap.error("provide sequences or --fasta")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = score(seqs, device)
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "hc50_uM", "broad_success_rate", "broad_mic50_uM", "safety_window"])
            for r in rows:
                p = r["panel"]
                w.writerow([r["sequence"], f"{r['hc50_uM']:.3g}",
                            f"{p['broad']['success_rate']:.3f}", f"{p['broad']['mic50']:.3g}",
                            f"{p.get('selectivity', {}).get('safety_window', float('nan')):.3g}"])
        print(f"wrote {args.out} ({len(rows)} rows)")
    else:
        for r in rows:
            p = r["panel"]
            print(f"{r['sequence']:30s} HC50={r['hc50_uM']:.3g}uM  "
                  f"broad_SR={p['broad']['success_rate']:.2f}  MIC50={p['broad']['mic50']:.3g}uM  "
                  f"SW={p.get('selectivity', {}).get('safety_window', float('nan')):.3g}")


if __name__ == "__main__":
    main()
