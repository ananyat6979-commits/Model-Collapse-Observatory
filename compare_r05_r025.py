# compare_r05_r025.py
import json
import numpy as np
from scipy import stats

with open('phase3_measurements/results/all_measurements_merged.json') as f:
    r05 = {r['generation_k']: r for r in json.load(f)}

with open('phase3_measurements/results/all_measurements_r025.json') as f:
    r025 = {r['generation_k']: r for r in json.load(f)}

key = 'perplexity_inversion_fixed'

print('PPL Inversion: R=0.5 vs R=0.25')
print(f'  Gen   R=0.5    R=0.25   Diff    Interpretation')
for k in [0, 1, 2, 3]:
    p05  = r05[k]['perplexity_inversion'].get('perplexity_inversion_ratio', 0) if k in r05 else 0
    p025 = r025[k][key]['perplexity_inversion_ratio'] if k in r025 else 0
    diff = p05 - p025
    interp = 'higher contamination -> stronger signal' if diff > 0.05 else 'within noise'
    print(f'  G{k}    {p05:.3f}    {p025:.3f}    {diff:+.3f}   {interp}')

print()
r05_ratios  = r05[3]['perplexity_inversion'].get('ppl_ratios_per_sample', []) if 3 in r05 else []
r025_ratios = r025[3][key].get('ppl_ratios_per_sample', []) if 3 in r025 else []
if r05_ratios and r025_ratios:
    U, p = stats.mannwhitneyu(r05_ratios, r025_ratios, alternative='two-sided')
    print(f'G3: R=0.5 mean={np.mean(r05_ratios):.3f} vs R=0.25 mean={np.mean(r025_ratios):.3f}')
    print(f'Mann-Whitney p={p:.6f}  (significant if p < 0.01)')