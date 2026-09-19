# Kaggle Writeup — copy-paste fields

Each section below maps to a field in the Kaggle writeup form. Copy the block under each heading.

---

## Title  *(required to save)*

AMPGen — A Two-Stage Generative Pipeline for Antimicrobial Peptide Design

---

## Subtitle

An activity-blind ESM2 generator paired with a dual-endpoint (MIC + HC50) predictor and safety-gated top-100 selection.

---

## Submission Track

**Benchmark participation (Minimum).**
The submission also meets the *technical* Full-track requirements — MIT license, `uv` + `uv.lock`, pinned Python 3.11, fixed default seed (42), and a verified `uv run generate` entry point. To convert to the **Full (co-authorship) track**, the only remaining steps are (1) make the GitHub repository public and (2) grant read access to @RasmusML and @szymczakpau.

---

## Content

### TL;DR
AMPGen is a two-stage pipeline. **Stage 1** is an *activity-blind* generator — ESM2-35M domain-adapted to the full antimicrobial-peptide corpus (162,339 sequences) and used as a masked-language-model Gibbs sampler — which proposes natural-like, constraint-valid, novel peptides. **Stage 2** is a *dual-endpoint predictor* on the same frozen encoder (LoRA adapter): a 10-species MIC head (species AdaLN conditioning) and a human-RBC HC50 head. Predictions are aggregated over the 20-strain challenge panel and combined into a single safety-gated composite that ranks the 50,000-member library down to the top-100. The whole submission reproduces from one command, `uv run generate` (seed 42).

### Pipeline
`AMP corpus (162,339)` → **Stage-1 generator** (ESM2-35M masked-LM Gibbs, activity-blind) → **50,000 library** (8–50 aa, 20 standard AA, linear/free-termini, deduplicated, no exact match to the reference set) → **dual-endpoint predictor** (frozen ESM2 + LoRA; MIC 10-species AdaLN head + HC50 head) → **panel aggregation** (species → 20 strains: Success Rate / MIC50 / Safety Window) → **safety-gated composite + Levenshtein novelty gate** → **top-100**.

### Training data & filters
| Asset | Sequences | Measurements | Sources (union) |
|---|---|---|---|
| Generator corpus | 162,339 | — | AMPSphere + DRAMP + DBAASP + dbAMP + APD + 7,059 activity-validated peptides |
| MIC (10 species) | 13,123 | 40,874 | GRAMPA + ampbench-v0.7 master + AMPBench-MT |
| HC50 (human RBC) | 4,402 | 4,402 | ampbench master `hc50_human` + Hemolytik2 + DRAMP |

All sources are public/academic. Filters: 20 standard amino acids, length 8–50, linear with free termini, no chemical modifications, exact-dedup; MIC harmonized to log₁₀ µM (AMPBench-MT pMIC → 6 − pMIC), per-(sequence, species) median; **leak-free split** via CD-HIT at 80% identity with whole clusters assigned to train/test. Licensing note: AMPBench-MT carries a **non-commercial** research license; it can be dropped for a strictly-permissive release (the fair ablation shows it mainly helps the rare-species categories).

### Predictor performance (honest, leak-free CD-HIT-80% split)
| Endpoint | Honest ρ / r | Random-split ρ | Within-cluster ceiling | Assay-noise oracle |
|---|---|---|---|---|
| MIC | **0.520 / 0.536** | 0.624 | 0.789 | 0.92–0.98 |
| HC50 | **0.481 / 0.492** | 0.603 | 0.80–0.90 (lit.) / 0.837 (within-cluster) |

The predictor (v4) is the separate-mode configuration (species-rebalanced sampling + censoring-aware HC50 loss) on the full merged corpus, with two per-endpoint levers chosen from a controlled 5-arm ablation on one fixed split: conservative MIC de-noising (drop cross-source contradictions > 3 doubling dilutions) for the MIC head, and a listwise soft-Spearman ranking term for the HC50 head. Relative to the published field on comparable homology-controlled splits, these numbers are competitive; we also report a **dataset-specific noise ceiling** (`r_max = √(1 − σ_e²/Var(y))`, estimated from cross-source measurement collisions), which — to our knowledge — no prior AMP predictor paper reports, and which shows HC50 retains the larger remaining headroom.

### Top-100 selection
Endpoints are predicted separately, aggregated per challenge category, then combined into one transparent composite: standardized broad-spectrum Success Rate + standardized log Safety Window (HC50/MIC50), **hard-gated** so that a potent-but-hemolytic peptide (Safety Window < 1) cannot rank at the top. Walking the ranking, a candidate is kept only if its **Levenshtein ratio ≤ 0.80** to every reference antibacterial peptide; the first 100 form `top.fasta`. The full library additionally contains no sequence identical to any reference.

### Reproducibility
```bash
uv sync
uv run generate            # -> generate/library.fasta (50,000) + generate/top.fasta (100)
```
Fixed default seed 42; identical output on repeated runs. A CUDA GPU is used when available (the submitted library was generated on a single H100) and the code falls back to CPU. `scripts/local_verify.py` mirrors the official checks (size, alphabet, length, uniqueness, no exact-reference overlap, top-100 Levenshtein ≤ 0.80) — the submission passes all of them.

---

## Project Description

### Abstract

AMPGen is a reproducible two-stage generative pipeline for antimicrobial-peptide design. Stage 1 is an *activity-blind* generator — ESM2-35M domain-adapted to the full AMP corpus and used as a masked-LM Gibbs sampler — that proposes novel, constraint-valid peptides for the 50,000-member library. Stage 2 is a dual-endpoint predictor on the same frozen ESM2 encoder (LoRA adapter): a 10-species MIC head with species AdaLN conditioning and a human-RBC HC50 head. Per-endpoint predictions are aggregated over the 20-strain panel and combined into a single safety-gated composite that ranks the library to the top-100. Honest, leak-free (CD-HIT-80%) performance is MIC Spearman ρ 0.520 and HC50 ρ 0.481, reported against a self-computed dataset-specific noise ceiling. The whole submission reproduces from `uv run generate` (seed 42).

### Data description

All training data is public/academic (no proprietary data). **Generator corpus** (162,339 sequences): AMPSphere + DRAMP + DBAASP + dbAMP + APD + 7,059 activity-validated peptides. **MIC** (13,123 sequences / 40,874 measurements, 10 species): GRAMPA + ampbench-v0.7 master + AMPBench-MT. **HC50** (4,402 sequences, human RBC): ampbench master `hc50_human` + Hemolytik2 + DRAMP. *Computational filters:* 20 standard amino acids, length 8–50, linear with free termini, no chemical modifications, exact-dedup; MIC harmonized to log₁₀ µM (AMPBench-MT pMIC → 6 − pMIC), per-(sequence, species) median; conservative de-noising drops (sequence, species) pairs whose cross-source measurements disagree by > 3 doubling dilutions; leak-free CD-HIT-80% whole-cluster train/test split. *Manual intervention:* none — all curation is programmatic and reproducible from the repository scripts. *Licensing note:* AMPBench-MT carries a non-commercial research license and can be dropped for a strictly-permissive release.

### Top candidates selection procedure (for Wet-Lab Track only)

Each endpoint is predicted separately (10-species MIC + HC50), then panel-aggregated per category into Success Rate (fraction of strains with MIC ≤ 16 µM), MIC50/MIC90, and Safety Window (HC50/MIC50). Candidates are ranked by a single composite = standardized category Success Rate + standardized log Safety Window, **hard-gated** so any peptide with Safety Window < 1 is excluded (no potent-but-hemolytic hits). Walking the ranking, a candidate is kept only if its Levenshtein ratio ≤ 0.80 to every reference antibacterial peptide; the first 100 form the list. Per wet-lab category, the potency term is swapped for that category's Success Rate — **Broad-Spectrum** (all 20 strains), **Gram-Positive** (5), **Gram-Negative** (15), **MDR ESKAPE** (8-strain subset) — while **Optimal Selectivity** is driven by the Safety-Window term and gate. Our submitted list uses a broad-spectrum + selectivity general composite.

---

## Project Links

- **Submission repository (self-contained, `uv run generate`):** https://github.com/YuHsin-Chen1221/AMP-Challenge
- **Method / training repository (paper):** https://github.com/YuHsin-Chen1221/AMPGen

> Both are currently **private**. For the Minimum track, grant read access to **@RasmusML** and **@szymczakpau**; for the Full track, make the submission repository **public**.

---

## Project Files

- `library.fasta` — the 50,000-sequence designed library
- `top.fasta` — the ranked top-100 candidates
- `AMPGen_Challenge_Report.pdf` — full submission report (method, data-scale, metric definitions, v1→v4 progression, fair ablation, per-category performance, noise-ceiling / oracle-gap)
- `abstract.md`, `training_data_disclosure.md`, `top100_selection.md` — method abstract, data disclosure, and top-100 selection documentation

*(All are in the submission repository: FASTA under `generate/`, report under `report/`, docs under `deliverables/`.)*
