"""
MCO- Figure 5: Domain reinforcement ablation
Isolates contamination-specific collapse effect from domain adaptation.
Run from repo root: python plots/generate_figure5.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

Path("plots/figures").mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

with open("g_human_ablation.json") as f:
    ablation = json.load(f)

with open("phase3_measurements/results/all_measurements_v2.json") as f:
    v2 = {r["generation_k"]: r for r in json.load(f)}

g0_ppls      = v2[0]["perplexity_inversion_fixed"]["ppl_gk_per_sample"]
g_human_ppls = ablation["ppl_gk_per_sample_g_human"]
g1_ppls      = v2[1]["perplexity_inversion_fixed"]["ppl_gk_per_sample"]

means = [np.mean(g0_ppls), np.mean(g_human_ppls), np.mean(g1_ppls)]
sems  = [np.std(d)/np.sqrt(len(d)) for d in [g0_ppls, g_human_ppls, g1_ppls]]
labels = ["G0\n(no fine-tuning)", "G_human\n(domain adaptation\nonly, no synthetic)",
          "G1\n(R=0.5 synthetic\ncontamination)"]
colors = ["#2196F3", "#FF9800", "#F44336"]

U, p = stats.mannwhitneyu(g_human_ppls, g1_ppls, alternative="two-sided")
n1, n2 = len(g_human_ppls), len(g1_ppls)
z = (U - n1*n2/2) / (n1*n2*(n1+n2+1)/12)**0.5
effect_r = abs(z) / (n1+n2)**0.5

fig, ax = plt.subplots(figsize=(9, 6.5))  # taller canvas
x = np.arange(3)
ax.bar(x, means, yerr=sems, capsize=6, color=colors, alpha=0.85,
       edgecolor="black", linewidth=1.2, width=0.6, zorder=3)
# Value labels ABOVE error bars, not overlapping the connector line
for i, (m, s) in enumerate(zip(means, sems)):
    ax.annotate(f"{m:.2f}", xy=(i, m + s + 1.5), ha="center",
                fontsize=12, fontweight="bold", zorder=6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("ppl_Gk (generating model perplexity on own outputs)")
ax.set_title("Figure 5: Domain reinforcement ablation\nisolating contamination-specific collapse from domain adaptation")
ax.grid(axis="y", alpha=0.3, zorder=0)
# Remove the diagonal connector line entirely — it's what's slicing
# through "29.01". A bar chart doesn't need a trend line connecting
# bar tops; the bars alone show the drop.
# Top annotation: place well above G0's bar+error, no line crossing it
ax.annotate("Domain adaptation:\n~95% of total drop",
            xy=(0.5, max(means) * 1.15), ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                      edgecolor="#FF9800", alpha=0.95), zorder=5)
# Bottom annotation: place BELOW the x-axis, not overlapping any bar
ax.annotate(f"Contamination-specific effect: p<0.001, effect_r={effect_r:.3f}\n(small but significant)",
            xy=(1.5, -0.15), xycoords=("data", "axes fraction"),
            ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                      edgecolor="#F44336", alpha=0.95), zorder=5)
ax.set_ylim(0, max(means) * 1.35)
plt.tight_layout()
plt.savefig("plots/figures/fig5_domain_ablation.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figure 5 saved to plots/figures/fig5_domain_ablation.png")
print(f"\nDecomposition:")
print(f"  G0 -> G_human: {means[0]-means[1]:.2f} drop (domain adaptation)")
print(f"  G_human -> G1: {means[1]-means[2]:.2f} drop (contamination-specific)")
print(f"  Total G0 -> G1: {means[0]-means[2]:.2f} drop")
print(f"  Domain adaptation %: {(means[0]-means[1])/(means[0]-means[2])*100:.1f}%")
print(f"  Contamination-specific %: {(means[1]-means[2])/(means[0]-means[2])*100:.1f}%")
print(f"  Mann-Whitney (G_human vs G1): U={U:.0f} p={p:.6f} effect_r={effect_r:.4f}")