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
report/                 challenge submission report (PDF + LaTeX + figure)
src/ampgen/generate.py  entry point: build the 50k library + ranked top-100
src/ampgen/predict.py   load weights + score peptides -> per-species MIC, HC50, category scores
src/ampgen/             inference code (predictor, constraints, novelty, ranking, physchem)
metrics/aggregate.py    panel aggregation: per-species -> 20-strain panel -> 4 category scores
scripts/local_verify.py mirror of the official verify_submission.py checks
weights/                model weights (git-lfs): generator + predictor encoder + predictor heads
deliverables/           abstract, top-100 selection, training-data disclosure
data/antibacterial.fasta  organiser novelty reference (39,448 seqs)
```

## Reproduce the submission (official entry point)

```bash
uv sync                       # installs pinned deps from uv.lock (Python 3.11)
uv run generate               # -> generate/library.fasta (50,000) + generate/top.fasta (100)
uv run generate --n-sequences 50000 --top-k 100 --seed 42   # defaults shown; all args optional
python scripts/local_verify.py generate/library.fasta generate/top.fasta   # PASS = submission-ready
```

Seed defaults to 42 (identical output on repeated runs). A CUDA GPU is used when available and the
code falls back to CPU; the full 50k run is GPU-bound (single H100 used for the submitted library).

## Score arbitrary peptides

```bash
uv run python -m ampgen.predict "KRWWKWWRRLLKK" "GLFDIVKKVVGALGSL"
uv run python -m ampgen.predict --fasta library.fasta --out scores.csv
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
