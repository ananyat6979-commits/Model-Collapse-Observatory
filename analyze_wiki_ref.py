import json
import numpy as np
from scipy import stats

with open("phase3_measurements/results/ppl_inversion_wiki_ref.json") as f:
    wiki = json.load(f)

with open("phase3_measurements/results/all_measurements_merged.json") as f:
    original = {r["generation_k"]: r for r in json.load(f)}

print("="*70)
print("DOMAIN GAP CONTROL, G0_wiki reference vs G0_original reference")
print("="*70)
print()
print("With G0_original (WebText: domain gap):")
for k in [1, 2, 3]:
    r = original.get(k, {}).get("perplexity_inversion", {})
    print(f"  G{k}: ratio={r.get('perplexity_inversion_ratio', 'N/A'):.4f}")

print()
print("With G0_wiki (Wikipedia-adapted: domain matched):")
for k in [1, 2, 3]:
    r = wiki.get(f"G{k}_r05", {})
    print(f"  G{k}: ratio={r.get('ppl_ratio', 'N/A'):.4f}")

print()
print("R=0.5 vs R=0.25 WITH domain-matched reference:")
print(f"  {'Gen':<6} {'R=0.5':>10} {'R=0.25':>10} {'Direction':>20}")
for k in [1, 2, 3]:
    r05  = wiki.get(f"G{k}_r05",  {}).get("ppl_ratio", 0)
    r025 = wiki.get(f"G{k}_r025", {}).get("ppl_ratio", 0)
    direction = "✓ R=0.5 > R=0.25" if r05 > r025 else "✗ still inverted"
    print(f"  G{k}     {r05:>10.4f} {r025:>10.4f} {direction:>20}")

# Mann-Whitney
r05_ratios  = wiki.get("G3_r05",  {}).get("ratios_per_sample", [])
r025_ratios = wiki.get("G3_r025", {}).get("ratios_per_sample", [])
if r05_ratios and r025_ratios:
    U, p = stats.mannwhitneyu(r05_ratios, r025_ratios, alternative="two-sided")
    print()
    print(f"G3 Mann-Whitney (R=0.5 vs R=0.25 with wiki ref):")
    print(f"  p={p:.6f}  {'significant' if p < 0.01 else 'not significant'}")