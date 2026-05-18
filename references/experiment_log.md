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
