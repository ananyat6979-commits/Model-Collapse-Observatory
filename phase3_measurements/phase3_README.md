# Phase 3: Measurement Layer Implementation

## Module Interface

Every module accepts:
```python
measure(generated_samples: list[str], baseline_pack: dict, **kwargs) -> dict[str, float]
```

No module imports from another module. Each is fully standalone.

## Self-Tests

Run all four on your laptop before any Kaggle run:
```powershell
python phase3_measurements\layers\lexical.py
python phase3_measurements\layers\semantic.py
python phase3_measurements\layers\tail_mass.py
python phase3_measurements\layers\perplexity_inversion.py
```

All four should print `PASSED`.

## Layer Summary

| Layer | File | Key Metrics | Self-Tests |
|---|---|---|---|
| Lexical | `lexical.py` | TTR, entropy, KL(gen‖baseline), Zipf | 6/6 |
| Semantic | `semantic.py` | cosine distance, coverage, intrinsic dim | 4/4 |
| Tail Mass | `tail_mass.py` | tail_mass_fraction, mean_log_likelihood | 3/3 |
| Perplexity Inversion | `perplexity_inversion.py` | PPL_G0/PPL_Gk ratio | 3/3 |

## Key Design Decisions

**Frozen encoder:** Loaded with `model.eval()` and `requires_grad=False`. An
assertion verifies this at measurement time. If you see "CRITICAL: encoder has
trainable parameters", stop immediately — all measurements would be invalid.

**Frozen tail threshold:** 7.4914 (5th percentile of baseline log-likelihood,
set in Phase 1). Never changes between generations. Adjusting it would
invalidate cross-generation comparisons.

**KL direction:** `KL(generated ‖ baseline)` — how much does generated text
deviate from the human baseline? NOT `KL(baseline ‖ generated)` which gives
the wrong collapse signal (collapsed text happens to contain baseline words,
so that direction shows LOW divergence for collapsed text — wrong).

## Kaggle Run

Before running, upload these four files to the `mco-phase2-artifacts` dataset:
```
phase1_baseline/measurements/lexical_baseline.json
phase1_baseline/measurements/semantic_baseline.json
phase1_baseline/measurements/ppl_baseline.json
phase1_baseline/measurements/kl_baseline_distributions.json
```

Then create a new Kaggle notebook (GPU T4 x2, Internet ON), attach
`mco-phase2-artifacts` as input, and paste cells from
`kaggle_measurement_notebook.py` in order.

**Save Version after every generation cell (G0, G1, G2, G3).**

## Expected Results (Pre-Experiment Hypothesis)

| Signal | G0 | G1 | G2 | G3 | Direction |
|---|---|---|---|---|---|
| Lexical entropy (bits) | ~11.93 | ↓ small | ↓ | ↓ | Decreasing |
| KL divergence | ~0 | ↑ | ↑ | ↑ | Increasing |
| Tail mass fraction | ~0.05 | ↓ 5–15% | ↓ | ↓ | Decreasing (primary) |
| PPL inversion ratio | ~1.0 | >1.0 | >1.1 | >1.2 | Increasing |
| Semantic cosine dist | baseline | ↓ small | ↓ | ↓ | Decreasing |

## Results

After Kaggle run completes, download `all_measurements.json` from notebook
output and place it at `phase3_measurements/results/all_measurements.json`.
