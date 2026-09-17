# Training data & external databases disclosure

All training data derives from public, academic antimicrobial-peptide databases. No proprietary or
non-public data was used. Three assets are built (all constraint-filtered and split leak-free).

## Data assets
| Asset | Sequences | Measurements | Sources (union) |
|---|---|---|---|
| Generator corpus | 162,339 | — | AMPSphere v2022-03 T1+T2 (physchem-matched) + DRAMP + DBAASP + dbAMP + APD + 7,059 activity-validated peptides (from the MIC/HC50 sets below) |
| MIC (10 species) | 13,123 | 40,874 | GRAMPA + ampbench-v0.7 master (= DBAASP+QMAP+HemoPI2+Hemolytik+DRAMP) + AMPBench-MT |
| HC50 (human RBC) | 4,402 | 4,402 | ampbench master `hc50_human` + Hemolytik2 + DRAMP free-text (14.2% right-censored) |

## Databases & terms
| Source | Role | Public | Notes |
|---|---|---|---|
| AMPSphere v2022-03 | generator pretrain corpus | yes | CC-BY; T1/T2 quality tiers, physchem-matched to curated envelope |
| DRAMP 3.0 | curated positives + MIC/HC50 text | yes | academic use |
| DBAASP v3 | curated positives + MIC + hemolysis | yes | academic use (subsumed via ampbench master) |
| dbAMP | curated positives | yes | academic use |
| APD3 | curated positives | yes | academic use |
| GRAMPA | per-species MIC labels | yes | derived compilation, public |
| ampbench v0.7 master | MIC + HC50 (DBAASP/QMAP/HemoPI2/Hemolytik/DRAMP union) | yes | our prior curated benchmark |
| AMPBench-MT | rare-species MIC | yes | HuggingFace `ZihengZhou06/AMPBench-MT`, **non-commercial research license** |
| Hemolytik2 | human-RBC HC50 | yes | Raghava lab, academic use |
| **antibacterial.fasta** | novelty reference only (not training) | yes | provided by the organisers |

## Computational filters applied
- **Constraint filter** (`src/ampgen/constraints.py`): 20 standard AAs, length 8–50, linear, free
  termini, no chemical modifications, exact-dedup.
- **Physchem-envelope match** for the AMPSphere pretrain tier (keeps the 2.5–97.5% length/charge/
  GRAVY/µH band of the curated AMPs, so the prior is not pulled toward long ORF-like sequences).
- **Unit harmonization**: MIC to log₁₀ µM (AMPBench-MT pMIC → `6 − pMIC`); HC50 to log₁₀ µM.
- **Aggregation**: per-(sequence, species) median MIC; per-sequence median HC50.
- **Leak-free split**: CD-HIT at 80% identity, whole clusters assigned to train/test (no test
  sequence > 80% identical to any training sequence). The clustering is used only to draw the
  train/test boundary — no sequences are removed beyond exact dedup.

## Manual intervention
None. All curation is programmatic and reproducible from the scripts in the method repository.

## Licensing note (for public/Full release)
All sources are public and academic. AMPBench-MT carries a non-commercial research license; if a
strictly permissive redistribution is required for Full/co-authorship eligibility, the AMPBench-MT
rows can be dropped (the fair ablation shows they help mainly the rare-species categories, not the
overall pipeline's viability).
