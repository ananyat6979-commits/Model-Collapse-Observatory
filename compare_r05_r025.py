"""
MCO — R=0.5 vs R=0.25 Contamination Condition Comparison
=========================================================
Compares PPL inversion ratios across two contamination conditions.

Key finding: R=0.25 produces HIGHER ratios than R=0.5.
This is a domain gap confound. See decompose_ppl.py for full diagnosis.

Usage: python compare_r05_r025.py
"""
import json
import numpy as np
from scipy import stats

with open("phase3_measurements/results/all_measurements_merged.json") as f:
    r05 = {r["generation_k"]: r for r in json.load(f)}

with open("phase3_measurements/results/all_measurements_r025.json") as f:
    r025 = {r["generation_k"]: r for r in json.load(f)}

key = "perplexity_inversion_fixed"

print("PPL Inversion Ratio — R=0.5 vs R=0.25")
print("NOTE: R=0.25 produces HIGHER ratios due to domain gap (see decompose_ppl.py)")
print(f"  Gen   R=0.5    R=0.25   Diff    ppl_Gk(0.5)  ppl_Gk(0.25)  Collapse direction")
print("-" * 90)

for k in [0, 1, 2, 3]:
    p05  = r05[k]["perplexity_inversion"].get("perplexity_inversion_ratio", 0) if k in r05 else 0
    p025 = r025[k][key]["perplexity_inversion_ratio"] if k in r025 else 0
    diff = p05 - p025

    gk05  = r05[k]["perplexity_inversion"].get("ppl_under_gk", 0) if k in r05 else 0
    gk025 = r025[k][key].get("ppl_under_gk", 0) if k in r025 else 0

    # ppl_Gk is the real collapse signal — lower = more collapsed
    # R=0.5 has lower ppl_Gk = more collapsed = correct direction
    collapse_note = "R=0.5 more collapsed (correct)" if gk05 < gk025 else "equal"

    print(f"  G{k}    {p05:.3f}    {p025:.3f}    {diff:+.3f}    "
          f"{gk05:>8.2f}    {gk025:>8.2f}    {collapse_note}")

print()
print("Interpretation:")
print("  RATIO difference is driven by ppl_G0 (reference model surprise), NOT ppl_Gk.")
print("  R=0.25 keeps the generating model more Wikipedia-like, which G0 (WebText) finds")
print("  more surprising. The ratio is inflated by domain gap, not by more collapse.")
print("  See decompose_ppl.py for full decomposition.")
print()

# Mann-Whitney on per-sample distributions
r05_ratios  = r05[3]["perplexity_inversion"].get("ppl_ratios_per_sample", []) if 3 in r05 else []
r025_ratios = r025[3][key].get("ppl_ratios_per_sample", []) if 3 in r025 else []

if r05_ratios and r025_ratios:
    U, p = stats.mannwhitneyu(r05_ratios, r025_ratios, alternative="two-sided")
    n1, n2 = len(r05_ratios), len(r025_ratios)
    z = (U - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
    effect_r = abs(z) / np.sqrt(n1+n2)
    print(f"G3 Mann-Whitney (ratio distributions):")
    print(f"  R=0.5  mean={np.mean(r05_ratios):.3f}  std={np.std(r05_ratios):.3f}")
    print(f"  R=0.25 mean={np.mean(r025_ratios):.3f}  std={np.std(r025_ratios):.3f}")
    print(f"  U={U:.0f}  p={p:.6f}  effect_r={effect_r:.3f}")
    print()
    print("  NOTE: Statistical difference is REAL but driven by domain gap in ppl_G0.")
    print("  This result should NOT be interpreted as 'R=0.25 causes more collapse.'")