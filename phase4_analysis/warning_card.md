# Early Warning Card
## Model Collapse Observatory: Phase 4 Analysis

**Model:** DistilGPT-2 (82M parameters)  
**Corpus:** English Wikipedia, HuggingFace 20220301.en, 5,000 documents, ~600k tokens  
**Contamination schedule:** Fixed R=0.5 across k=3 generations  
**Measurement baseline:** Phase 1 Wikipedia corpus (same source as training human data)

---

## Signal Detection Order

| Rank | Signal | First Detection | Effect Size | G0→G3 Change | Monotonic |
|---|---|---|---|---|---|
| 1 | PPL Inversion Ratio | **k=1** | 2.52 | +172.5% | ✓ |
| 2 | Zipf Alpha | **k=1** | 2.62 | +24.8% | ✓ |
| 3 | Type-Token Ratio | **k=1** | 2.57 | -47.9% | ✓ |
| 4 | KL Divergence (1gram) | k=2 | 2.47 | +6.5% | ✓ |
| 5 | Entropy (1gram) | k=2 | 2.47 | -3.1% | ✓ |
| 6 | Tail Mass Fraction | not detected | 2.22 | +28.6%* | ✗ |
| 7 | Semantic Cosine Distance | not detected | 0.60 | -0.1% | ✗ |

*Tail mass moved in wrong direction: see caveats.

---

## Practical Implications

**If your training dataset is 50% synthetic (R=0.5), after one fine-tuning
generation you should expect:**

- **PPL inversion ratio ≈ 1.95** (up from 1.0). The model generates text
  it finds nearly twice as predictable as the reference model does.
  This is the earliest and largest detectable signal.

- **Type-token ratio drops ~31%** (0.112 → 0.077). Vocabulary is
  substantially more repetitive after a single contaminated generation.

- **Zipf alpha increases ~13%** (1.03 → 1.17). Word frequency distribution
  concentrates more heavily on common words: rare words disappear from output.

After three contaminated generations at R=0.5:

- **PPL inversion ratio ≈ 2.73**: generating model is >2.7x more confident
  about its outputs than the human-trained reference model.
- **TTR drops ~48%**: the model generates from roughly half the vocabulary
  it used at generation 0.
- **1-gram entropy decreases 3.1%**: modest but consistent.

---

## Actionable Thresholds

At contamination ratio R=0.5 (equal mix of human and synthetic):

| Signal | Threshold | Detects at |
|---|---|---|
| PPL inversion ratio > 1.5 | Reliable detection | k=1 |
| TTR drops > 20% from k=0 | Reliable detection | k=1 |
| Zipf alpha increases > 10% | Reliable detection | k=1 |
| KL divergence increases > 3% | Reliable detection | k=2 |
| Entropy drops > 2% | Reliable detection | k=2 |

**Recommended monitoring approach:** Track PPL inversion ratio at every generation.
A ratio consistently above 1.5 indicates collapse has begun. A ratio above 2.5
indicates severe concentration of generated distribution.

---

## Primary Collapse Signal: ppl_Gk (Revised)

Initial analysis proposed the PPL inversion ratio (ppl_ref / ppl_Gk) as the
primary signal. After the domain-matched reference experiment (Sprint 2), the
correct recommendation is:

**Use ppl_Gk, the generating model's perplexity on its own outputs, as the
primary collapse signal.**

ppl_Gk values across conditions:

| Gen | R=0.5 ppl_Gk | R=0.25 ppl_Gk | Correct order |
|---|---|---|---|
| G0 | 29.01 | 29.01 | baseline |
| G1 | 4.31 | 4.89 | ✓ R=0.5 lower |
| G2 | 2.60 | 3.23 | ✓ R=0.5 lower |
| G3 | 1.92 | 2.65 | ✓ R=0.5 lower |

ppl_Gk correctly orders contamination severity at every generation.
R=0.5 produces more collapsed models (lower ppl_Gk) than R=0.25, as expected.

**Why the ratio fails across conditions:**
As collapse deepens, the generating model concentrates on high-frequency patterns
that are predictable to any model including the reference. ppl_ref drops alongside
ppl_Gk. The ratio (ppl_ref / ppl_Gk) does not increase monotonically with
contamination severity because both components decrease.

Within a single condition (fixed R, increasing k), the ratio is still a valid
diagnostic. It tracks the within-condition progression correctly. The problem
arises only when comparing across contamination ratios.

**Actionable threshold (revised):**
Monitor ppl_Gk directly. Values below 5.0 indicate significant collapse has begun.
Values below 2.5 indicate severe concentration of the generation distribution.

---

## ZL-IRL Boundary Positioning

These signals are effective within the **stochastic contamination regime**,
partial synthetic data mixing at fixed ratio. This study operates in the
pre-boundary regime where early warning is possible.

Under deterministic representation truncation (the ZL-IRL boundary condition),
early warning becomes structurally impossible. The signals characterized here
detect the *approach* to that boundary, they are not valid at or beyond it.

*Reference: Zero-Lead-Time Irreversible Representation Loss (ZL-IRL),
ananyat6979-commits/Zero-Lead-Time-Irreversible-Representation-Loss.*

---

## Caveats and Limitations

**Scale constraints:**
- Results calibrated at DistilGPT-2 (82M) scale only. Detection thresholds
  may not generalize to larger models.
- Corpus: 5,000 documents, ~600k tokens. Small corpus may amplify collapse.
- k=3 generations. Longer collapse chains may exhibit different dynamics.
- One contamination ratio (R=0.5) tested. Behavior at R=0.1 or R=0.9 unknown.

**Domain constraint:**
- Corpus and measurement baseline are English Wikipedia.
- Generating model (DistilGPT-2) was originally trained on WebText (Reddit),
  creating a domain mismatch. Signals were measured in Wikipedia space, but
  the model's outputs were never fully in that space.

**Tail mass finding:**
- Tail mass fraction moved in the *opposite* direction to prediction (+28.6%,
  not the predicted decrease). Root cause: G0 outputs already fall in the
  tail of the Wikipedia KDE (G0 tail fraction = 0.159, vs baseline 0.028).
  The Wikipedia density estimator is an incorrect reference for a WebText-trained
  model. Tail mass requires domain-matched baseline to function as designed.
  **For future work:** compute tail mass relative to the G0 output distribution,
  not the human baseline distribution.

**Semantic coverage:**
- Returned 0.0 for all generations. GMM components (n=10) fitted on 20-dim PCA
  embeddings were too sparse for the 5k document synthetic corpus. Not meaningful
  at this scale.

**Statistical testing:**
- With 4 generations and point estimates (no per-sample distributions for most
  metrics), formal p-value testing is not feasible. Effect sizes and monotonicity
  are used as proxies. Phase 5 at larger scale would enable proper hypothesis testing.

---

## Measurement Code

All four measurement layers available at:
`phase3_measurements/layers/`

Interface:
```python
result = measure(generated_samples, baseline_pack, **kwargs)
```

Reference pack (baseline measurements) available at:
`phase1_baseline/reference_pack.pkl`

Results database:
`phase3_measurements/results/all_measurements.json`

To reproduce: see `phase3_measurements/kaggle_measurement_notebook.py`

---

## Next Steps

1. **Fix tail mass:** Use G0 output embeddings as the density reference instead
   of Phase 1 Wikipedia. Re-run tail mass measurements with corrected reference.

2. **Increase k:** Run k=5 or k=7 generations to observe whether PPL inversion
   ratio plateaus or continues increasing. Current data suggests it may plateau
   between k=2 and k=3 (+2.61 → +2.73 is a small increment).

3. **Vary R:** Test at R=0.2 and R=0.8 to determine minimum contamination ratio
   at which each signal becomes detectable.

4. **Scale model:** Repeat with a 1B parameter model (LLaMA-3.2-1B) to verify
   whether PPL inversion signal generalizes beyond DistilGPT-2.
