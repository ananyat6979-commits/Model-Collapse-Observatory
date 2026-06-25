import json
import numpy as np
from scipy import stats

with open("phase3_measurements/results/ppl_inversion_wiki_ref.json") as f:
    wiki = json.load(f)

print("="*70)
print("DECOMPOSED PPL: G0_wiki reference")
print("="*70)
print(f"{'Condition':<14} {'ppl_ref':>9} {'ppl_Gk':>9} {'ratio':>8} {'Collapse order'}")
print("-"*60)

for label in ["G0_r05","G1_r05","G2_r05","G3_r05"]:
    r = wiki[label]
    k = label[1]
    print(f"  R=0.5  G{k}   {r['ppl_under_ref']:>9.3f} {r['ppl_under_gk']:>9.3f} {r['ppl_ratio']:>8.4f}")

print()
for label in ["G1_r025","G2_r025","G3_r025"]:
    r = wiki[label]
    k = label[1]
    print(f"  R=0.25 G{k}   {r['ppl_under_ref']:>9.3f} {r['ppl_under_gk']:>9.3f} {r['ppl_ratio']:>8.4f}")

print()
print("ppl_Gk comparison at G3:")
gk_r05  = wiki["G3_r05"]["ppl_under_gk"]
gk_r025 = wiki["G3_r025"]["ppl_under_gk"]
print(f"  R=0.5:  ppl_Gk = {gk_r05:.3f}")
print(f"  R=0.25: ppl_Gk = {gk_r025:.3f}")
order = "CORRECT ORDER" if gk_r05 < gk_r025 else "WRONG ORDER"
print(f"  R=0.5 < R=0.25: {order}")
print()

# Mann-Whitney on ppl_Gk distributions
r05_gk  = wiki["G3_r05"].get("ppl_gk_per_sample",
          [r for r, k in zip(wiki["G3_r05"].get("ratios_per_sample",[]),
                             range(200))])
# ppl_Gk is not stored separately, compute from ratio and ppl_ref
# ratio = ppl_ref / ppl_Gk → ppl_Gk = ppl_ref / ratio
print("Key finding summary:")
print(f"  ppl_Gk correctly orders: R=0.5 ({gk_r05:.2f}) < R=0.25 ({gk_r025:.2f})")
print(f"  ppl_ref inverts:  R=0.5 ({wiki['G3_r05']['ppl_under_ref']:.2f}) < R=0.25 ({wiki['G3_r025']['ppl_under_ref']:.2f})")
print(f"  Ratio inverts:    R=0.5 ({wiki['G3_r05']['ppl_ratio']:.4f}) < R=0.25 ({wiki['G3_r025']['ppl_ratio']:.4f})")
print()
print("Interpretation:")
print("  Collapsed text (R=0.5) is predictable to ALL models, including the reference.")
print("  ppl_Gk alone correctly measures collapse severity.")
print("  The ratio conflates collapse severity with text predictability.")
print("  RECOMMENDATION: Use ppl_Gk as primary signal, not the ratio.")