# MCO Contamination Index — Public Schema v1.0

This document defines the schema for contamination index entries,
the minimum reproducibility bar for new submissions, and the methodology
for interpreting scores.

---

## Index Entry Schema

```json
{
  "dataset_name": "string — canonical HuggingFace dataset ID or descriptive name",
  "dataset_version": "string — version tag or date",
  "dataset_source_url": "string — publicly accessible URL",
  "ground_truth": "string — known/unknown/partial + description",
  "sample_size": "integer — number of documents scored",
  "sample_strategy": "string — how documents were sampled",
  "measurement_date": "string — YYYY-MM-DD",
  "encoder_id": "string — HuggingFace model ID of frozen reference encoder",
  "reference_model": "string — model used for PPL inversion measurement",

  "scores": {
    "ttr": "float — type-token ratio of sampled documents",
    "entropy_1gram": "float — unigram Shannon entropy (bits)",
    "kl_div_1gram": "float — KL(generated || Wikipedia baseline) in bits",
    "avg_pairwise_cos_dist": "float — avg cosine distance in encoder space",
    "semantic_coverage": "float — NN coverage vs Wikipedia baseline embeddings",
    "mean_ppl_under_g0": "float — mean perplexity under reference model",
    "ppl_ratio_vs_baseline": "float — baseline_ppl / dataset_ppl",
    "composite_contamination_index": "float — weighted composite, 0-1"
  },

  "component_scores": {
    "lexical_ttr": "float — 0-1, normalized TTR contamination signal",
    "lexical_kl": "float — 0-1, normalized KL divergence signal",
    "ppl_predictability": "float — 0-1, normalized PPL ratio signal",
    "semantic_coverage": "float — 0-1, inverted coverage signal"
  },

  "signal_weights_used": {
    "lexical_ttr": 0.865,
    "lexical_kl": 0.865,
    "ppl_predictability": 0.865,
    "semantic_coverage": 0.300
  },

  "estimated_synthetic_fraction": {
    "point_estimate": "float — composite index value (see caveats)",
    "ci_lower_95": "float",
    "ci_upper_95": "float",
    "estimation_method": "string"
  },

  "caveats": ["array of strings — required"],
  "full_measurements": "object — raw layer outputs for reproducibility"
}
```

---

## What "Composite Contamination Index" Means

The composite index is a **normalized signal strength**, not a calibrated probability.

A value of 0.0 means the dataset's statistical properties are identical to the
Wikipedia human baseline used to calibrate this framework.

A value of 1.0 means the dataset's statistical properties match the most collapsed
generation (k=3, R=0.5, DistilGPT-2) from the Phase 3 simulation.

**It does NOT mean X% of the documents are synthetic.** It means the distribution of
the sampled documents has shifted X fraction of the way from "Wikipedia-like" to
"fully collapsed." A dataset with many short, simple documents may score high
without containing any AI-generated content. A dataset with carefully curated
synthetic content that mimics human writing may score low.

The composite index is an **anomaly detector**, not a classifier.

---

## Calibration Anchors

All scores are normalized against Phase 3 simulation results:

| Anchor | TTR | KL div | PPL ratio | Source |
|---|---|---|---|---|
| G0 (k=0, clean) | 0.1123 | 5.057 | 1.000 | Phase 3, DistilGPT-2 |
| G3 (k=3, R=0.5) | 0.0585 | 5.383 | 2.725 | Phase 3, DistilGPT-2 |
| Wikipedia holdout | expected < 0.20 | — | — | Negative control |
| TinyStories | 0.667 | — | — | Known 100% synthetic |

**Signal weights** are derived from Phase 4 Mann-Whitney effect sizes:
- lexical_ttr, lexical_kl, ppl_predictability: weight=0.865 (large effect, U=0)
- semantic_coverage: weight=0.300 (weak signal at 5k document scale)
- tail mass: **excluded** — domain-dependent, not portable across domains

---

## Minimum Reproducibility Requirements for New Submissions

To submit a new dataset entry to the contamination index:

1. **Public dataset access.** The dataset must be downloadable from a public URL
   without authentication. HuggingFace Hub datasets are preferred.

2. **Fixed sample.** The submission must specify the exact sampling procedure
   (dataset split, seed, n documents) such that anyone can reproduce the same sample.

3. **Reference pack version.** The measurement must use the Phase 1 reference pack
   at commit hash `d7e2d6c6...` (or a clearly documented alternative baseline).

4. **Encoder must be frozen.** The sentence-transformers encoder
   `all-MiniLM-L6-v2` must be used in eval mode with `requires_grad=False`.
   Using a different encoder produces incomparable results.

5. **Four measurements required.** All four layers (lexical, semantic, tail mass
   with domain-matched baseline, PPL) must be reported even if some are excluded
   from the composite index.

6. **Caveats are mandatory.** Every entry must include at minimum:
   - Calibration model and scale
   - Encoder identity
   - Sample size and strategy
   - Statement that composite index ≠ synthetic fraction

7. **No cherry-picking.** If a dataset is scored, all results must be reported
   regardless of direction. Selective reporting undermines the index's utility.

---

## Limitations and Scope

This index is calibrated against:
- **Model:** DistilGPT-2 (82M parameters)
- **Domain:** English Wikipedia
- **Scale:** 5,000 training documents, 135 tokens each
- **Language:** English only
- **Contamination type:** Token/document-level mixing at fixed ratios

Results **may not generalize** to:
- Models larger than ~1B parameters (different PPL distribution)
- Non-English datasets
- Code, mathematics, or technical domain datasets
- Contamination patterns other than uniform random mixing
- Datasets where human and synthetic text are semantically similar

---

## Ethical Considerations

Publishing contamination scores for datasets produced by specific organizations
requires care:

1. **Report scores, not verdicts.** A high composite index means "statistically
   different from Wikipedia human text" — not "this organization used AI."
   Many legitimate causes can raise the score (domain mismatch, writing style,
   topic distribution).

2. **Always include caveats.** Every entry must state the limitations of the
   measurement. Do not publish a score without its uncertainty.

3. **Do not name organizations in the index key.** Name the dataset, not the
   organization. "pile_cc" not "EleutherAI_data."

4. **Threshold for publication.** Do not publish an index entry for a dataset
   unless the negative control (Wikipedia holdout) has passed (composite < 0.25).
   A failed negative control means the framework is not calibrated correctly for
   the current setup and scores would be misleading.

5. **Version all results.** If the framework is updated (new calibration, new
   reference model), scores from previous versions are not comparable. Version
   the index entries alongside the framework version.

---

## Comparison Table (as of Phase 5 pilot)

| Dataset | Composite | TTR | PPL ratio | Ground truth | Notes |
|---|---|---|---|---|---|
| Wikipedia holdout (20%) | PENDING | — | — | Known human | Negative control |
| TinyStories | 0.667 | 0.061 | 2.143 | 100% GPT-4 | Validation passed |
| C4 validation | PENDING | — | — | Unknown | Expected 0.15-0.35 |
| Pile-CC subset | PENDING | — | — | Unknown | Expected 0.20-0.45 |

---

## Reproduce This Index

```bash
# Clone the repository
git clone https://github.com/ananyat6979-commits/Model-Collapse-Observatory

# Install dependencies
pip install -r requirements.txt

# Score a dataset
python phase5_index/run_dataset.py --dataset wikipedia_holdout
python phase5_index/run_dataset.py --dataset c4
python phase5_index/run_dataset.py --dataset pile_cc
```

Reference pack download:
```python
# The reference pack is committed to the repository at:
# phase1_baseline/reference_pack_manifest.json
# SHA-256: d7e2d6c611aa7b9a47f628a328c0729488229515470bf209aec31404fe4905f1
# Re-generate from source: python phase1_baseline/run.py --seed 42 --test-only
```
