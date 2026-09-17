# Top-100 selection & ranking procedure

## Objective
**Broad-spectrum + optimal-selectivity (general)** — we do not lock a single competition category.
The ranking rewards peptides that are both potent across the whole 20-strain panel and selective
(non-hemolytic), so a candidate is strong across the activity categories *and* the Optimal-
Selectivity category simultaneously. (To instead target one category, swap the potency term in the
composite for that category's Success Rate: `gram_pos` / `gram_neg` / `mdr`.)

## Pipeline (fully reproducible, `uv run generate`, seed 42)
1. **Generate** the 50,000-member library — activity-blind masked-LM Gibbs sampler, generator's own
   length distribution, hard 8–50 gate; filtered to 20 AAs / linear / free-termini / unique, and no
   sequence identical to any reference in `data/antibacterial.fasta`.
2. **Predict, per sequence, each endpoint separately** (debuggable, category-decomposable): the
   10-species MIC head + the HC50 head (`src/ampgen/predict.py`).
3. **Panel aggregation** (`src/ampgen/aggregate.py`): broadcast per-species MIC to the 20 panel
   strains (via `PANEL_STRAIN_COUNTS`) → per-category **Success Rate** (MIC ≤ 16 µM), **MIC50/MIC90**,
   and **Safety Window** = HC50 / MIC50 (challenge ceilings applied: MIC 64 µM, HC50 128 µM).
4. **Composite score** (`src/ampgen/ranking.py`), one scalar per sequence:

   ```
   composite = z(broad Success Rate) + z(log10 Safety Window)          # standardized across the 50k
   composite = -inf   if Safety Window < 1.0                            # safety gate: no toxic-but-potent
   ```
5. **Rank** by the composite, best-first.
6. **Novelty gate (strict)**: walking the ranking, keep a sequence only if its **Levenshtein ratio
   ≤ 0.80** to every reference in `data/antibacterial.fasta`; stop at 100. Output → `generate/top.fasta`.

## Why a single composite (not separate reports or Pareto)
Endpoints are predicted and reported separately (transparent, category-decomposable), but the final
selection needs one ranked list, so they are combined into a single transparent composite. The
safety **gate** (not a soft weight) guarantees no potent-but-hemolytic peptide reaches the top —
directly serving the Optimal-Selectivity criterion — while the standardized sum balances potency and
selectivity on equal footing. A Pareto front was rejected: it cannot land exactly 100 and is harder
to document.

## Predictor honest performance (context for the ranking's trust)
Leak-free split: MIC Spearman ρ 0.585 / Pearson r 0.597 (RMSE 0.65 log₁₀ µM); HC50 ρ 0.503 / r 0.522.
Ranking is the reliable signal; absolute MIC/HC50 are calibrated but noisier, so the composite leans
on rank-order (Success-Rate fraction and Safety-Window ordering) more than on exact µM values.
