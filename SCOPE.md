# SCOPE.md — Model Collapse Observatory

---

## What This Project Is

The Model Collapse Observatory (MCO) is a controlled, small-scale experimental system
for measuring distributional collapse in LLMs trained on synthetic data.

**Core claim:** Distributional collapse from synthetic data contamination is measurable
before it becomes visible on standard benchmarks, using a four-layer signal hierarchy.

**Novel contribution:** The perplexity inversion ratio and the systematized tail mass
measurement framework — applied together to form the first public contamination index
for real training datasets.

---

## What This Project Is NOT

This is not:
- A production-scale LLM training system
- A study of catastrophic collapse (full failure of generation quality)
- A benchmark of state-of-the-art language models
- A generalization of findings beyond the experimental parameters below

All claims from this project are scoped strictly to the experimental parameters
in this file. Any paper, preprint, or public artifact must restate these limits.

---

## Experimental Parameters (Locked)

These are immutable after Phase 2 begins. Any change requires starting over from Phase 1.

```yaml
model:               distilgpt2           # 82M parameters — only supported model
corpus_source:       Wikipedia (frozen dump) + BookCorpus subset
corpus_url:          https://dumps.wikimedia.org/enwiki/20241201/
                     # FILL IN: exact dump filename used
corpus_sha256:       FILL_IN_AFTER_DOWNLOAD
corpus_size_docs:    5000
corpus_size_tokens:  ~500000
corpus_language:     English only
encoder:             all-MiniLM-L6-v2     # frozen permanently from Phase 1
                     # CPU-friendly; correct choice for this compute budget
                     # Results NOT comparable across different encoder choices
generations_k:       3                   # G0, G1, G2, G3
contamination_ratio: 0.5                 # fixed across all generations
contamination_schedule: fixed            # same ratio every generation (not R^k)
seed:                42
platform_phase1:     Laptop (CPU only)
platform_phase2:     Kaggle T4
platform_phase3:     Laptop (dev) + Kaggle T4 (runs)
platform_phase4:     Laptop (CPU only)
platform_phase5:     Laptop (CPU only) — TinyStories pilot
```

---

## Why These Parameters

**DistilGPT-2 (82M):** Fits in Kaggle T4 VRAM with room for the reference model
simultaneously (required for perplexity inversion). Fast enough to run k=3
fine-tuning cycles in one Kaggle session. Large enough to exhibit real distributional
collapse dynamics.

**5k documents / ~500k tokens:** Sufficient to fit a non-degenerate KDE and PCA.
Zipf alpha is stable across subsamples at this scale. Embedding computation is
feasible on CPU in <15 minutes with all-MiniLM-L6-v2.

**R=0.5 fixed:** Chosen to isolate the accumulation effect across generations.
At each generation, the model sees the same synthetic/human ratio, but the synthetic
data is increasingly degraded. This cleanly separates "collapse from contamination
fraction" from "collapse from accumulated degradation."

**all-MiniLM-L6-v2:** Produces 384-dimensional embeddings vs 768 for all-mpnet-base-v2.
Approximately 3× faster on CPU. Sufficient for the intrinsic dimensionality and
semantic coverage measurements at this scale.

**Common Crawl explicitly excluded:** CC tail is contaminated with HTML artifacts,
encoding errors, and boilerplate. These create false positives in tail mass measurements
(rare-but-valid vs rare-and-broken). Phase 5 contamination index entries against
CC-derived datasets must note this explicitly.

---

## Relationship to ZL-IRL

This project is complementary to the Zero-Lead-Time Irreversible Representation Loss
(ZL-IRL) work, which establishes:

> In a memoryless self-training system with deterministic representation truncation,
> the system transitions directly SAFE → HIGH_RISK with no detectable WARNING regime.
> Early warning is structurally impossible in this class of systems.
> [ananyat6979-commits/Zero-Lead-Time-Irreversible-Representation-Loss]

MCO operates in the **pre-boundary regime**:
- Stochastic contamination (not deterministic truncation)
- Partial synthetic data mixing (not 100% synthetic feedback)
- Continuous embedding spaces (not vocabulary truncation)

Within this regime, MCO's hypothesis is that early warning IS possible.
MCO's signals are detectors for the APPROACH to the ZL-IRL boundary.
Once deterministic truncation occurs and tail mass reaches zero, detection fails —
consistent with ZL-IRL's finding.

**This framing must appear in:** SCOPE.md (here), any related work section, the
Phase 4 Early Warning Card, and all Phase 5 contamination index documentation.

---

## Compute Constraints and Platform Allocation

| Phase | Platform | Rationale |
|-------|----------|-----------|
| 1: Baseline | Laptop (CPU only) | Embedding 5k docs with MiniLM takes ~10 min CPU |
| 2: Simulation | Kaggle T4 | Fine-tuning DistilGPT-2 requires GPU |
| 3: Measurement | Laptop (dev) + Kaggle (runs) | Modules on laptop; scale runs on Kaggle |
| 4: Analysis | Laptop (CPU only) | KS/Mann-Whitney on SQLite rows — no GPU needed |
| 5: Pilot | Laptop (CPU only) | TinyStories 1k sample — CPU sufficient |

Colab: debugging and testing only (<20 min sessions). Never for production runs.
HuggingFace Spaces: Streamlit dashboard hosting.

---

## Publication Strategy

**Option A — Workshop paper (most achievable in 1–2 months):**
Phase 4 Early Warning Card as a short paper for NeurIPS/ICML workshop on synthetic
data quality. Framing: "Preliminary empirical evidence for early warning signals in
the stochastic contamination regime, scoped to DistilGPT-2 at small scale."

**Option B — Datasets Track (requires Phase 5 expansion):**
Contamination index with ≥3 datasets submitted to NeurIPS Datasets and Benchmarks.

**Option C — Preprint:**
All five phases as an arXiv preprint. No venue-imposed scale constraints.

Do not try to combine all phases into a single venue submission at this scale.

---

## Honest Limitations

Every paper or preprint from this project must state all of the following:

- Results are calibrated at DistilGPT-2 scale (82M). Detection thresholds may not
  generalize to larger models.
- Corpus is English Wikipedia + BookCorpus. Domain-specific collapse may differ.
- Only one contamination ratio (R=0.5) and one schedule (fixed) are tested.
  Detection order may differ at other ratios.
- k=3 generations is a small simulation. Longer collapse chains may behave differently.
- The perplexity inversion signal is novel and unvalidated by prior work. Treat Phase 4
  results for this signal as preliminary evidence, not established fact.
- Phase 5 baseline is Wikipedia/BookCorpus, not Common Crawl. Contamination index
  scores for CC-derived datasets reflect deviation from a Wikipedia-style reference,
  not an internet-text reference.
