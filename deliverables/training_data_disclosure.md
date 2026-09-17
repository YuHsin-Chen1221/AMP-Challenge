# Training data & external databases disclosure (DRAFT)

<!-- Required minimum deliverable. Must list: training data, external DBs used, and any
manual intervention or computational filters. For Full/co-authorship: any non-public data
must be released publicly under a permissive license. -->

## Databases used
| Source | Role | License / terms | Public? |
|--------|------|-----------------|---------|
| DRAMP 3.0 | curated positives | __ | yes |
| DBAASP | curated positives + MIC | check terms | yes |
| dbAMP | curated positives | __ | yes |
| APD | curated positives | __ | yes |
| AMPSphere v2022-03 | pretrain corpus (T1+T2) | __ | yes |
| GRAMPA / ConformAMP v0.7 | MIC labels | __ | yes |
| (HC50 source — TBD, e.g. DBAASP hemolysis / HemoPI) | hemolysis labels | __ | __ |

## Computational filters applied
- Constraint filter (`src/ampgen/constraints.py`): 20 AA, len 8–50, linear, free termini, dedup.
- Physchem-envelope match for the pretrain corpus (2.5–97.5% len/charge/GRAVY/µH).
- CD-HIT 80% whole-cluster train/test split (leakage control).

## Manual intervention
- __ (list any hand-curation)

TODO: fill license columns; confirm every source is redistributable under a permissive
license before public release.
