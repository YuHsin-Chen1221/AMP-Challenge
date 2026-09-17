# Method abstract

**AMPGen** is a two-stage generative pipeline for antimicrobial peptide design. **Stage 1** is an
activity-blind generator: ESM2-35M domain-adapted to the full antimicrobial-peptide corpus
(162,339 sequences) with a variable-rate span/scattered masking curriculum, then used as a
masked-language-model Gibbs sampler to produce novel, natural-like linear peptides. Generation
follows the generator's own learned length distribution under a hard 8–50-residue gate; sequences
are filtered for the challenge constraints (20 standard amino acids, linear, free termini),
deduplicated, and screened against the reference set to yield the 50,000-member library.

**Stage 2** is a dual-endpoint predictor built on the same frozen encoder (LoRA adapter, r=8, on the
last four attention layers): a **10-species MIC head** with species AdaLN conditioning predicting
absolute −log₁₀ MIC, and a **human-RBC HC50 head** predicting log₁₀ HC50, trained separately with a
composite ranking + calibrated-magnitude loss (censoring-aware for the right-censored HC50 tail).
Predictions are aggregated over the 20-strain panel (species → strains via panel strain counts) into
per-category Success Rate, MIC50/90 and Safety Window (HC50/MIC50).

**Top-100 selection** ranks the library by a single, transparent composite — standardized
broad-spectrum Success Rate plus standardized log Safety Window, gated so a potent-but-hemolytic
peptide (Safety Window < 1) cannot rank at the top — and enforces the strict novelty rule
(Levenshtein ratio ≤ 0.80 to any reference antibacterial peptide).

On a leak-free CD-HIT-80% whole-cluster split, the predictor reaches honest MIC Spearman ρ 0.585 /
Pearson r 0.597 and HC50 ρ 0.503 / r 0.522; a controlled data ablation confirms the merged training
corpus improves ranking most on the resistance-relevant MDR and Gram categories. The pipeline is
fully reproducible from a fixed random seed via `uv run generate`.
