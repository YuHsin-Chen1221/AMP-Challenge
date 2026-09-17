# Top-100 selection & ranking procedure (DRAFT)

<!-- Required minimum deliverable: document how the top-100 candidates were selected and
ranked from the 50k library. -->

## Target category
- [ ] Broad-Spectrum / [ ] Gram-Positive / [ ] Gram-Negative / [ ] MDR ESKAPE / [ ] Optimal Selectivity
  (decide — drives the ranking objective; Optimal Selectivity additionally needs the HC50 head)

## Pipeline
1. Generate 50k constraint-valid library (`generate/`, fixed seed).
2. Score with the species-conditioned MIC predictor (`../src/ampgen/predictor.py`).
3. (Optional) optimize with annealed MH-MCMC (`../scripts/optimize_loop.py`).
4. Enforce novelty: ≤80% identity to any DBAASP/dbAMP/APD peptide.
5. Rank by __ ; take top 100.

TODO: finalize objective + produce top100.csv.
