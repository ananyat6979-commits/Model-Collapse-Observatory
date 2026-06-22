# MCO Experiment Log

All experiments must have hypothesis written BEFORE running.
Format: EXP-NNN | Phase | Date | Hypothesis | Result | Supported

---

## EXP-000 - Kaggle Persistence Verification
**Phase:** 2 (pre-experiment)
**Date:** 2026-04-08
**Hypothesis:** Files written to Kaggle Dataset in one session will be accessible
from a new session by mounting the dataset as input.
**Setup:** Write dummy file to mco-phase2-artifacts dataset, disconnect, reconnect,
verify file visible at /kaggle/input/mco-phase2-artifacts/.
**Result:** PASSED. persistence_test.txt and persistence_test_v2.txt both visible
in dataset after session reconnect. Reference pack and documents.jsonl also confirmed
accessible via input mount.
**Hypothesis supported:** YES
**Next step:** Begin generation loop. G0 must be saved before any fine-tuning.

---

## EXP-001 - Phase 2 Generation k=1 (G1)
**Phase:** 2
**Date:** 2026-04-09
**Hypothesis:** At generation k=1 with R=0.5, the model (G1 fine-tuned on H + S0)
will show early-regime collapse signals. Expected training loss lower than G0's
pretraining baseline on this corpus (~44 PPL equivalent), because G1 is adapting
to a partially synthetic distribution. The synthetic outputs S1 from G1 will be
less diverse than S0 from G0 — measurable in Phase 3.
**Setup:**
- Model: distilgpt2 (82M)
- Training corpus: 5000 human docs + 5000 G0 synthetic docs (R=0.5)
- Fine-tuning steps: 3000
- LR: 5e-5, batch size: 8 per GPU × 2 GPUs, fp16
- Seed: 42
**Result:** Final training loss: 2.5996 (recovered from trainer_state.json step 3000).
G1 checkpoint saved. S1 (5000 synthetic docs) generated and saved.
**Hypothesis supported:** PENDING — Phase 3 measurements needed to verify diversity claim.
**Next step:** G2 fine-tuning.

---

## EXP-002 - Phase 2 Generation k=2 (G2)
**Phase:** 2
**Date:** 2026-04-10
**Hypothesis:** G2 trained on H + S1 (R=0.5) will show lower training loss than G1
(2.5996), as the model increasingly fits a concentrated synthetic distribution.
The loss decrease from G1→G2 should be larger than G2→G3 (diminishing returns).
**Setup:**
- Training corpus: 5000 human docs + 5000 G1 synthetic docs (R=0.5)
- Fine-tuning from G1_checkpoint. Steps: 3000, seed: 42
**Result:** Final loss: 1.989. Loss drop G1→G2: 0.61 nats.
**Hypothesis supported:** PARTIALLY — loss did decrease (confirmed). Whether G1→G2
drop > G2→G3 drop: need G3 result to verify.
**Next step:** G3 fine-tuning.

---

## EXP-003 - Phase 2 Generation k=3 (G3)
**Phase:** 2
**Date:** 2026-04-10
**Hypothesis:** G3 trained on H + S2 (R=0.5) will show lower loss than G2 (1.989),
but the decrease will be smaller than G1→G2 (diminishing returns on collapse).
**Setup:**
- Training corpus: 5000 human docs + 5000 G2 synthetic docs (R=0.5)
- Fine-tuning from G2_checkpoint. Steps: 3000, seed: 42
**Result:** Final loss: 1.659. Loss drop G2→G3: 0.330 nats.
**Hypothesis supported:** YES — G1→G2 drop (0.61) > G2→G3 drop (0.33). Diminishing
returns confirmed. Loss curve: 2.60 → 1.99 → 1.66.
**Observation:** Loss trajectory consistent with early-regime collapse. Not catastrophic.
Model is in the stochastic contamination regime, not at the ZL-IRL boundary.
**Next step:** Phase 3 measurements on all generations.

---

## EXP-004 - Phase 3 Measurement Layer Self-Tests
**Phase:** 3
**Date:** 2026-04-11
**Hypothesis:** All four measurement modules will produce correct directional results
on synthetic known-collapse inputs: collapsed text (identical strings) should score
lower diversity than diverse text across all layers.
**Setup:** Each module run with:
  - Collapsed case: identical repeated strings
  - Diverse case: random unique-word strings
**Result:**
  - lexical.py: 5/5 PASS (KL direction bug fixed: KL(gen||baseline) not KL(baseline||gen))
  - semantic.py: 4/4 PASS
  - tail_mass.py: 3/3 PASS
  - perplexity_inversion.py: 3/3 PASS
**Hypothesis supported:** YES, all four layers correctly identify collapse direction.
**Bug found:** lexical.py originally computed KL(baseline||generated). This gives low
KL for collapsed text (collapsed generates "the" which IS in baseline). Fixed to
KL(generated||baseline): collapsed text (all "the") now correctly gives high KL=3.47
because gen_p("the")≈1 >> baseline_p("the")≈0.05.
**Next step:** Upload Phase 1 measurement files to Kaggle. Run measurement notebook
on all four generations. Record results into measurements.db.

---

## Pending Experiments

### EXP-005 (PLANNED) - Phase 3 Full Measurement Run
**Phase:** 3
**Pre-experiment hypothesis:**
At k=3 with R=0.5 fixed (DistilGPT-2, 5k docs, 3 generations):
- Lexical entropy: small decrease 2–8% from baseline (11.93 bits)
- Tail mass fraction: 10–25% decrease from baseline ~5% (primary signal)
- PPL inversion ratio: 1.1–1.4 at k=3 (generated text increasingly predictable
  to itself, increasingly strange to G0 reference)
- Semantic cosine distance: small decrease 2–5%
Tail mass should show the largest relative change. PPL inversion should fire
before semantic diversity. Lexical is the most conservative signal.
**Falsified if:** Any signal shows no monotonic trend G0→G3, or tail mass fraction
increases at any generation.
**Status:** PENDING: waiting for Kaggle run.

---

## EXP-005 - Phase 3 Full Measurement Run (All Four Layers, G0–G3)
**Phase:** 3
**Date:** 2026-05-17
**Pre-experiment hypothesis (EXP-005 from previous log):**
At k=3 with R=0.5: tail_mass_fraction decreases 10–25%, PPL inversion ratio
reaches 1.1–1.4, lexical entropy small decrease 3–8%, semantic cosine dist
small decrease 2–5%.

**Setup:**
- Model: DistilGPT-2 (82M), corpus: HF Wikipedia 20220301.en, 5k docs
- Four measurement layers on G0-G3 synthetic outputs
- Platform: Kaggle T4 x2, ~20 min total runtime

**Results:**

| Signal | G0 | G1 | G2 | G3 | Predicted | Outcome |
|---|---|---|---|---|---|---|
| TTR | 0.112 | 0.077 | 0.064 | 0.058 | Decrease | ✓ Confirmed |
| entropy_1gram | 10.54 | 10.45 | 10.27 | 10.22 | Decrease | ✓ Confirmed |
| kl_div_1gram | 5.057 | 5.150 | 5.331 | 5.383 | Increase | ✓ Confirmed |
| zipf_alpha | 1.034 | 1.171 | 1.250 | 1.291 | Increase | ✓ Confirmed |
| avg_pairwise_cosine_dist | 0.966 | 0.969 | 0.965 | 0.965 | Decrease | ✗ Flat |
| tail_mass_fraction | 0.159 | 0.209 | 0.204 | 0.204 | Decrease | ✗ Increased |
| ppl_inversion_ratio | 1.000 | 1.947 | 2.610 | 2.725 | 1.1–1.4 | ✓ Larger than expected |

**Hypothesis supported:** PARTIALLY
- Lexical signals: all confirmed, all monotonic
- PPL inversion: confirmed and exceeded expectation (+172.5%, predicted +10–40%)
- Tail mass: FALSIFIED- went up not down (root cause documented below)
- Semantic: below noise level at this scale

**Root cause analysis: tail mass anomaly:**
G0 tail_mass_fraction = 0.159, but Phase 1 baseline tail_fraction = 0.028.
G0 outputs are already 5.6x more "tail-like" than the human Wikipedia corpus.
DistilGPT-2 was trained on WebText (Reddit links) — it generates text in a
different semantic register from Wikipedia. When projected onto the Wikipedia
KDE, G0 outputs land in low-density regions by default, not because of collapse.
As fine-tuning proceeds on a mixed corpus, outputs do not move toward the
Wikipedia density center: they stay in the same off-distribution region.
DIAGNOSIS: The tail mass measurement requires a domain-matched baseline. Using
the Wikipedia KDE as reference for a WebText-trained model is incorrect.
FIX: Fit the KDE on G0 outputs, not Phase 1 Wikipedia. Tail threshold = 5th
percentile of G0 log-likelihood. Then tail mass measures deviation from G0's
own distribution, not the human corpus.

**Novel finding: PPL inversion:**
ppl_under_gk: 29.0 → 4.3 → 2.6 → 1.9 (near-degenerate by k=3)
ppl_under_g0: 29.0 → 8.2 → 6.7 → 5.3 (human ref model still "surprised")
Ratio: 1.0 → 1.95 → 2.61 → 2.73
This is the cleanest signal in the study: monotonic, large effect, novel.
The model at k=3 generates text with effective PPL≈2, near-certainty about
its own outputs. This is the mechanistic signature of distributional collapse.

**Next step:**
Phase 4 statistical analysis. Fix tail mass measurement for re-run on Kaggle.
Write Early Warning Card.

---

## EXP-006 — Phase 4 Signal Priority Analysis (Point Estimates)
**Phase:** 4
**Date:** 2026-05-17
**Hypothesis:** PPL inversion fires before tail mass (H3), tail mass fires before
semantic (H1), semantic fires before lexical (H2). From EXP-005 predictions.

**Setup:** `phase4_analysis/early_warning.py` on all_measurements.json

**Results:**
Signal detection order (first generation with >2% deviation from G0):
1. PPL inversion ratio: k=1 (172.5% total change, effect=2.52, monotonic)
2. Zipf alpha: k=1 (24.8% change, effect=2.62, monotonic)
3. TTR: k=1 (-47.9% change, effect=2.57, monotonic)
4. KL divergence: k=2 (6.5% change, effect=2.47, monotonic)
5. Entropy 1gram: k=2 (-3.1% change, effect=2.47, monotonic)
6. Tail mass: not detected (anomalous direction, non-monotonic)
7. Semantic cosine: not detected (0.1% change, non-monotonic)

**Hypothesis test results:**
- H1 (tail fires before semantic): REJECTED (neither fires reliably)
- H2 (semantic fires before lexical): REJECTED (lexical fires, semantic doesn't)
- H3 (PPL fires before tail mass): SUPPORTED (PPL fires at k=1, tail never)

**Interpretation:**
H3 is the novel claim: PPL inversion fires earliest. SUPPORTED.
H1 and H2 were based on collapse theory that doesn't account for domain mismatch.
At this scale and with this domain mismatch, lexical signals are more reliable
than the density-based signals (tail mass, semantic).
The lexical signals (TTR, Zipf, KL) are simpler to compute and fired correctly.
The density-based signals require domain alignment to work as designed.

**Hypothesis supported:** H3 supported; H1, H2 rejected due to domain mismatch
in density measurements, not due to theoretical failure.

**Next step:**
Fix tail mass to use G0-relative density. Run Phase 5 with TinyStories pilot.
Consider also running a corrected Phase 3 measurement pass with fixed tail mass.

---

## EXP-006: Phase 2 Generation at R=0.25 (Lower Contamination Condition)
**Phase:** 2 (second experimental condition)
**Date:** 2026-05-19
**Hypothesis:** At R=0.25 (75% human, 25% synthetic), collapse should proceed
more slowly than at R=0.5. Training loss should be higher at every generation
(model fits the contaminated distribution less tightly). PPL inversion signal
should be weaker at every k.
**Setup:**
- Model: distilgpt2 (82M), same G0 checkpoint as R=0.5 experiment
- Contamination ratio: R=0.25 (fixed)
- G0 outputs (S0) reused from R=0.5 experiment — same seed, same model
- Training: 3000 steps per generation, same hyperparameters as R=0.5
- Seed: 42

**Training loss progression:**

| Generation | R=0.5 loss | R=0.25 loss | Difference |
|---|---|---|---|
| G1 | 2.5996 | 2.6995 | +0.100 |
| G2 | 1.9892 | 2.2302 | +0.241 |
| G3 | 1.6592 | 1.9764 | +0.317 |

Loss gap widens with each generation: R=0.25 model overfits less to the
contaminated distribution at each step. Consistent with hypothesis.

**PPL inversion results (UNEXPECTED):**

| Gen | R=0.5 ratio | R=0.25 ratio | Difference |
|---|---|---|---|
| G0 | 1.000 | 1.000 | 0.000 |
| G1 | 1.947 | 2.341 | -0.395 |
| G2 | 2.610 | 3.260 | -0.651 |
| G3 | 2.725 | 3.726 | -1.001 |

Mann-Whitney G3 R=0.5 vs R=0.25: p<0.001

**Hypothesis supported:** PARTIALLY
- Training loss: SUPPORTED, R=0.25 shows consistently higher loss (weaker overfit)
- PPL inversion direction: FALSIFIED, R=0.25 shows HIGHER ratio, not lower

**Root cause of PPL reversal (domain gap confound):**
G0 reference model (DistilGPT-2) was trained on WebText (Reddit links).
Training corpus is English Wikipedia. These are different distributions.

At R=0.25 (75% Wikipedia in training mix): fine-tuned model generates more
Wikipedia-like text. G0 (WebText) finds Wikipedia-style outputs MORE surprising,
PPL_G0 increases, ratio is higher.

At R=0.5 (50% Wikipedia): model generates more mixed outputs. Some synthetic
content resembles WebText (G0's training domain), reducing PPL_G0, reducing ratio.

The PPL inversion ratio measures distributional distance from G0's training
distribution, not collapse severity directly. Domain alignment between reference
model and training corpus is required for unconfounded collapse measurement.

**Next step:** Document domain alignment requirement as a hard constraint in
SKILL.md and warning_card.md. For future work: use G0 trained on the same domain
as the training corpus (e.g., a Wikipedia-trained model as reference for
Wikipedia experiments).

---

## EXP-007: Phase 5 Contamination Index: Three Real Datasets
**Phase:** 5
**Date:** 2026-05-20
**Hypothesis:** The MCO measurement framework will produce high composite index
scores for synthetic datasets and near-zero scores for human datasets.
Specifically: TinyStories (100% GPT-4) > 0.5, Wikipedia holdout < 0.2,
C4 validation 0.15-0.35, Pile-CC 0.20-0.45.

**Results:**

| Dataset | TTR | KL | PPL ratio | Composite | Ground truth |
|---|---|---|---|---|---|
| Wikipedia holdout | 0.227 | 4.15 | 0.905 | 0.026 | Known human |
| C4 validation | 0.185 | 4.03 | 0.631 | 0.023 | Human web |
| Pile-CC | 0.221 | 3.50 | 0.834 | 0.016 | Human web |
| TinyStories | 0.061 | 6.14 | 2.143 | 0.667 | 100% GPT-4 |

**Hypothesis supported:** PARTIALLY
- TinyStories: SUPPORTED — composite 0.667 > 0.5 ✓
- Wikipedia holdout: SUPPORTED — composite 0.026 < 0.2 ✓ (negative control passes)
- C4 and Pile-CC: FALSIFIED — both score near zero, not 0.15-0.45 as predicted

**Root cause of C4 and Pile-CC scoring near zero:**
The composite index is calibrated against DistilGPT-2 output space.
Human web corpora (C4, Pile-CC) have higher TTR (0.18-0.22) and more diverse
vocabulary than DistilGPT-2 outputs (G0 TTR=0.112, G3 TTR=0.058).

The entire lexical calibration range (G0 TTR to G3 TTR) lies below the
diversity level of real web text. C4 and Pile-CC are MORE diverse than the
clean-generation anchor, so they floor at zero on the lexical component.

PPL ratio for C4 (0.63) and Pile-CC (0.83) are both below 1.0, G0 finds
web text HARDER to predict than Wikipedia, consistent with web text being more
diverse and out-of-distribution for a WebText-trained model. Neither dataset
registers as synthetic on the PPL component.

**Key finding: framework scope:**
The MCO framework detects distributional collapse in the DistilGPT-2 output
space. It correctly flags text that is MORE collapsed than DistilGPT-2 outputs
(TinyStories: simple GPT-4 children's stories). It cannot detect text that is
more diverse than DistilGPT-2 outputs (C4, Pile-CC: real web text).

**What "contamination" the framework measures:**
Not "fraction of synthetic documents" but "how close is this text to the
collapsed distribution of DistilGPT-2 at k=3, R=0.5."

**To extend to web-domain datasets:**
Replace Phase 1 baseline corpus with C4 or Common Crawl samples.
Re-fit all measurements (KDE, PCA, KL distributions) on web-domain text.
Re-calibrate composite index against web-domain collapse simulation.
This is a 2-3 month project and is documented as future work.

**Next step:** Phase 5 is complete as a pilot. Document calibration limitation
in INDEX_SCHEMA.md and warning_card.md. Mark Phase 5 signed off.
The measurement framework is validated for its stated scope.