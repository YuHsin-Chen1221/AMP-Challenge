"""Stage-2 MIC predictor (proposal §2.2): species-conditioned scorer on the shared encoder.

Design (locked 2026-09-11):
  - Encoder = the Stage-1 all-AMP generator checkpoint loaded as EsmModel (hidden 480),
    FROZEN, with LoRA (peft) on the last 4 of 12 layers' attention query/key/value only.
    This is the "same encoder as the generator" with a pluggable predictor adapter.
  - Masked mean-pool over residue tokens -> E (480-d).
  - Species AdaLN-Zero: gamma,beta from a zero-init Linear(species_emb); E_cond =
    E*(1+gamma)+beta (identity at init, so the predictor starts species-agnostic and
    learns the conditioning). Primary conditioning key = target species (E.coli/S.aureus/
    P.aeruginosa) — the highest-leverage causal key for MIC.
  - MLP head -> scalar score s(seq, species). Higher score = more active (lower MIC).

Trained with a pairwise margin-ranking loss on within-species pairs (train_predictor.py):
s orders sequences by activity, so Delta = s(MT) - s(WT) is the closed-loop DeltaMIC signal,
and Spearman rho of s vs -log10(MIC) is the eval. Epistasis is captured by ESM attention +
the ranking objective; the explicit Hadamard interaction tensor (proposal §3.1) is deferred.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Canonical species keys for the MIC head = the 10 species that make up the challenge's
# 20-strain panel (every panel strain maps to exactly one of these). GRAMPA has >=300
# measurements for each. Order is FIXED and append-only: the species Embedding is indexed by
# SPECIES_TO_ID, so appending (never reordering) keeps earlier 3-species checkpoints loadable.
SPECIES = [
    "e. coli", "s. aureus", "p. aeruginosa",          # original 3 (kept first for continuity)
    "k. pneumoniae", "a. baumannii", "e. faecalis", "e. faecium",
    "b. subtilis", "s. enterica", "e. cloacae",
]
SPECIES_TO_ID = {s: i for i, s in enumerate(SPECIES)}

# Gram stain per species -> the Gram-positive / Gram-negative competition categories.
GRAM = {
    "e. coli": "neg", "p. aeruginosa": "neg", "a. baumannii": "neg",
    "k. pneumoniae": "neg", "s. enterica": "neg", "e. cloacae": "neg",
    "s. aureus": "pos", "b. subtilis": "pos", "e. faecalis": "pos", "e. faecium": "pos",
}

# How many of the 20 official panel strains each species represents. Used to aggregate a
# per-species prediction up to the strain-counted panel metrics (Success Rate, MIC50/90).
# Gram- (15): E.coli x5, P.aeruginosa x3, A.baumannii x2, K.pneumoniae x2, S.enterica x2,
#             E.cloacae x1.   Gram+ (5): S.aureus x2, B.subtilis x1, E.faecalis x1, E.faecium x1.
PANEL_STRAIN_COUNTS = {
    "e. coli": 5, "p. aeruginosa": 3, "a. baumannii": 2, "k. pneumoniae": 2,
    "s. enterica": 2, "e. cloacae": 1, "s. aureus": 2, "b. subtilis": 1,
    "e. faecalis": 1, "e. faecium": 1,
}  # sums to 20

# The 8-strain MDR ESKAPE subset, counted by species (E. coli contributes AIC222 + BAA-3170).
MDR_SUBSET_COUNTS = {
    "a. baumannii": 1, "e. coli": 2, "k. pneumoniae": 1, "p. aeruginosa": 1,
    "s. aureus": 1, "e. faecalis": 1, "e. faecium": 1,
}  # sums to 8


class SpeciesConditionedScorer(nn.Module):
    def __init__(self, encoder_ckpt: str, n_species: int = len(SPECIES),
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 lora_layers=(8, 9, 10, 11), hidden: int | None = None,
                 pool: str = "mean", head_dims=(None,)):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        enc = AutoModel.from_pretrained(encoder_ckpt)
        d = hidden or enc.config.hidden_size
        for p in enc.parameters():                      # freeze base; LoRA adds trainable
            p.requires_grad = False
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         bias="none", target_modules=["query", "key", "value"],
                         layers_to_transform=list(lora_layers), layers_pattern="layer")
        self.encoder = get_peft_model(enc, cfg)

        self.pool = pool
        if pool == "attn":                              # learned-query attention pooling
            self.attn_q = nn.Parameter(torch.randn(d) * 0.02)
        self.species_emb = nn.Embedding(n_species, d)
        self.adaln = nn.Linear(d, 2 * d)                # -> gamma, beta
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)  # AdaLN-Zero
        self.norm = nn.LayerNorm(d)
        # MLP head: head_dims=(None,) -> single hidden layer of d//2 (the v1 default);
        # e.g. (256, 64) -> two hidden layers 480->256->64->1.
        def _mlp():
            dims = [d] + [(d // 2 if h is None else h) for h in head_dims] + [1]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers += [nn.GELU(), nn.Dropout(0.1)]
            return nn.Sequential(*layers)

        self.head = _mlp()          # MIC head (fed the species-AdaLN-modulated embedding)
        # Hemolysis (HC50) head: SAME architecture, SAME shared encoder + pooled embedding, but
        # deliberately NO species AdaLN -- hemolysis is a single human-RBC endpoint, not
        # species-conditioned. This is the "unified encoder, different head" design.
        self.hemo_head = _mlp()
        self.d = d

    def _pool(self, input_ids, attention_mask, residue_mask=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = (residue_mask if residue_mask is not None else attention_mask).float()  # [B, L]
        if self.pool == "attn":
            s = (h @ self.attn_q) / (self.d ** 0.5)      # [B, L] attention logits
            s = s.masked_fill(mask == 0, -1e9)
            w = torch.softmax(s, dim=1).unsqueeze(-1)     # [B, L, 1]
            return (h * w).sum(1)                         # attention-weighted pool
        m = mask.unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean pool -> [B, d]

    def _score(self, E, species_id):
        gamma, beta = self.adaln(self.species_emb(species_id)).chunk(2, dim=-1)
        E_cond = self.norm(E) * (1 + gamma) + beta       # identity at init (zero-init adaln)
        return self.head(E_cond).squeeze(-1)             # [B] score

    def forward(self, input_ids, attention_mask, species_id, residue_mask=None):
        E = self._pool(input_ids, attention_mask, residue_mask)
        return self._score(E, species_id)

    def forward_hemo(self, input_ids, attention_mask, residue_mask=None):
        """Hemolysis score, calibrated to raw log10 HC50 (uM); higher = safer / less hemolytic.
        Shares the encoder + pooling with the MIC head; skips species AdaLN (single endpoint).
        log Safety Window = HC50/MIC50 -> forward_hemo + forward add (both "good" directions)."""
        E = self._pool(input_ids, attention_mask, residue_mask)
        return self.hemo_head(self.norm(E)).squeeze(-1)

    def forward_all(self, input_ids, attention_mask, residue_mask=None):
        """Multi-output cross-species head: score the SAME pooled embedding under every
        species' AdaLN modulation in one pass -> [B, n_species]. Lets a sequence with
        labels in several species supervise several outputs jointly (shared trunk + head),
        which is the channel that transfers the ~0.8 cross-species MIC correlation."""
        E = self._pool(input_ids, attention_mask, residue_mask)          # [B, d]
        n = self.species_emb.num_embeddings
        cols = [self._score(E, torch.full((E.shape[0],), s, device=E.device, dtype=torch.long))
                for s in range(n)]
        return torch.stack(cols, dim=1)                                  # [B, n_species]
