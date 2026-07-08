import json
import numpy as np
from scipy import stats

with open("g_human_ablation.json") as f:
    ablation = json.load(f)

with open("phase3_measurements/results/all_measurements_v2.json") as f:
    v2 = {r["generation_k"]: r for r in json.load(f)}

g_human_ppls = ablation["ppl_gk_per_sample_g_human"]
g1_ppls      = v2[1]["perplexity_inversion_fixed"]["ppl_gk_per_sample"]

print(f"G_human: n={len(g_human_ppls)}  mean={np.mean(g_human_ppls):.3f}  std={np.std(g_human_ppls):.3f}")
print(f"G1:      n={len(g1_ppls)}  mean={np.mean(g1_ppls):.3f}  std={np.std(g1_ppls):.3f}")

U, p = stats.mannwhitneyu(g_human_ppls, g1_ppls, alternative="two-sided")
n1, n2 = len(g_human_ppls), len(g1_ppls)
z = (U - n1*n2/2) / (n1*n2*(n1+n2+1)/12)**0.5
effect_r = abs(z) / (n1+n2)**0.5

print(f"\nMann-Whitney U={U:.0f}  p={p:.6f}  effect_r={effect_r:.4f}")

if p < 0.05:
    print("\nCONTAMINATION EFFECT IS STATISTICALLY SIGNIFICANT")
    print("beyond domain adaptation, though small in magnitude.")
else:
    print("\nCONTAMINATION EFFECT IS NOT DISTINGUISHABLE FROM NOISE")
    print("at this sample size. The G0->G1 drop is essentially all")
    print("domain adaptation. This is a much stronger and more honest")
    print("finding than the original framing.")