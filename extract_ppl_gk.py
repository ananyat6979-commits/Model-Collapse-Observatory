import json
import numpy as np
from scipy import stats

with open("phase3_measurements/results/all_measurements_v2.json") as f:
    v2 = {r["generation_k"]: r for r in json.load(f)}

with open("phase3_measurements/results/all_measurements_r025.json") as f:
    r025 = {r["generation_k"]: r for r in json.load(f)}

def get_ppl_gk(record, key="perplexity_inversion_fixed"):
    ppl = record.get(key, record.get("perplexity_inversion", {}))
    if "ppl_gk_per_sample" in ppl:
        return ppl["ppl_gk_per_sample"]
    # Compute from ratio and ppl_ref: ppl_gk = ppl_ref / ratio
    ratios = ppl.get("ppl_ratios_per_sample", [])
    refs   = ppl.get("ppl_g0_per_sample", [])
    if ratios and refs:
        return [round(r/ratio, 4) for r, ratio in zip(refs, ratios) if ratio > 0]
    return []

print("ppl_Gk cross-condition comparison at G3:")
gk_r05  = get_ppl_gk(v2[3])
gk_r025 = get_ppl_gk(r025[3], "perplexity_inversion_fixed")

print(f"  R=0.5  G3 ppl_Gk: mean={np.mean(gk_r05):.3f}  n={len(gk_r05)}")
print(f"  R=0.25 G3 ppl_Gk: mean={np.mean(gk_r025):.3f}  n={len(gk_r025)}")

if gk_r05 and gk_r025:
    U, p = stats.mannwhitneyu(gk_r05, gk_r025, alternative="two-sided")
    n1, n2 = len(gk_r05), len(gk_r025)
    z = (U - n1*n2/2) / (n1*n2*(n1+n2+1)/12)**0.5
    effect_r = abs(z) / (n1+n2)**0.5
    print(f"  Mann-Whitney: p={p:.6f}  effect_r={effect_r:.3f}")
    print(f"  {'R=0.5 < R=0.25 (correct)' if np.mean(gk_r05) < np.mean(gk_r025) else 'WRONG DIRECTION'}")