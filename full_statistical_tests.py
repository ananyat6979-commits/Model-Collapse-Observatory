import json
import numpy as np
from scipy import stats

with open("phase3_measurements/results/lexical_per_doc.json") as f:
    lex = {int(k): v for k, v in json.load(f).items()}

with open("phase3_measurements/results/all_measurements_v2.json") as f:
    ppl_data = {r["generation_k"]: r for r in json.load(f)}

alpha_bonf = 0.01 / (3 * 3)  # 3 signals × 3 generations = 9 tests

print("="*65)
print("FULL STATISTICAL TESTS — All Layers")
print(f"Bonferroni threshold: p < {alpha_bonf:.4f}")
print("="*65)

g0_ttr = lex[0]["ttr_per_doc"]
g0_kl  = lex[0]["kl_per_doc"]
g0_ppl = ppl_data[0]["perplexity_inversion_fixed"]["ppl_ratios_per_sample"]

print(f"\n{'Signal':<12} {'G0 mean':>9} {'Gk mean':>9} {'U':>8} {'p':>10} {'effect_r':>9} {'sig':>5}")
print("-"*65)

for k in [1, 2, 3]:
    for signal, g0_vals, gk_key, gk_sub in [
        ("TTR",  g0_ttr, "ttr_per_doc", None),
        ("KL",   g0_kl,  "kl_per_doc",  None),
        ("PPL",  g0_ppl, "ppl_ratios_per_sample", "perplexity_inversion_fixed"),
    ]:
        if gk_sub:
            gk_vals = ppl_data[k][gk_sub][gk_key]
        else:
            gk_vals = lex[k][gk_key]

        U, p = stats.mannwhitneyu(g0_vals, gk_vals, alternative="two-sided")
        n1, n2 = len(g0_vals), len(gk_vals)
        z = (U - n1*n2/2) / np.sqrt(n1*n2*(n1+n2+1)/12)
        effect_r = abs(z) / np.sqrt(n1+n2)
        sig = "✓" if p < alpha_bonf else "✗"
        print(f"G{k} {signal:<8} {np.mean(g0_vals):>9.4f} {np.mean(gk_vals):>9.4f} "
              f"{U:>8.0f} {p:>10.2e} {effect_r:>9.3f} {sig:>5}")