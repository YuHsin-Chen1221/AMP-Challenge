# AMP-Challenge — AMPGen submission

Submission for the **NeurIPS 2026 Competition Track AMP Challenge** (szczurek-lab). A two-stage
generative pipeline for antimicrobial peptide design:

- **Stage-1 generator** — a domain-adapted ESM2-35M masked-LM (Gibbs mutation proposer) that
  produces natural-like candidate peptides (`weights/generator/`).
- **Stage-2 dual-endpoint predictor** — per-species MIC (10 species) + human-RBC HC50 heads on the
  frozen encoder (`weights/predictor_mic.pt`, `weights/predictor_hemo.pt`), scoring and ranking the
  library for activity and selectivity.

This repo is **self-contained** for inference and scoring; the full training methodology and
ablations live in the companion method repo (AMPGen).

## Layout

```
report/              challenge submission report (PDF + LaTeX + figure)
src/ampgen/          minimal inference code (predictor, constraints, physchem)
metrics/aggregate.py panel aggregation: per-species -> 20-strain panel -> 4 category scores
scripts/predict.py   load weights + score peptides -> per-species MIC, HC50, category scores
weights/             model weights (git-lfs): generator encoder + predictor heads
deliverables/        abstract, top-100 selection, training-data disclosure
```

## Quick start

```bash
uv venv && uv pip install torch transformers peft pandas numpy scipy   # or pip
python scripts/predict.py "KRWWKWWRRLLKK" "GLFDIVKKVVGALGSL"
python scripts/predict.py --fasta library.fasta --out scores.csv
```

## Model

Recommended predictor: **v2 configuration** (separate mode; rare-species sampling rebalance +
censoring-aware HC50 loss), trained on the full merged corpus for deployment. Honest (homology-split)
metrics from the held-out evaluation runs: MIC Spearman ρ 0.585 / Pearson r 0.597; HC50 ρ 0.503 /
r 0.522. See `report/AMPGen_Challenge_Report.pdf` for the full data-scale, metric definitions,
v1→v3 progression and per-category performance.

## Data

MIC: GRAMPA ∪ ampbench-v0.7 master ∪ AMPBench-MT (13,123 seqs / 40,874 measurements, 10 species).
HC50: ampbench master `hc50_human` ∪ Hemolytik2 ∪ DRAMP (4,402 seqs). Leak-free CD-HIT-80%
whole-cluster split. Full disclosure in `deliverables/training_data_disclosure.md`.
