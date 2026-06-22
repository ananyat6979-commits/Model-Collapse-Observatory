import json
import numpy as np

with open('phase3_measurements/results/all_measurements_merged.json') as f:
    r05 = {r['generation_k']: r for r in json.load(f)}

with open('phase3_measurements/results/all_measurements_r025.json') as f:
    r025 = {r['generation_k']: r for r in json.load(f)}

key = 'perplexity_inversion_fixed'
print('Decomposed PPL values')
print(f'Gen  | ppl_G0(R0.5)  ppl_Gk(R0.5)  ratio(R0.5) | ppl_G0(R025)  ppl_Gk(R025)  ratio(R025)')
print('-' * 95)
for k in [0, 1, 2, 3]:
    if k not in r05 or k not in r025:
        continue
    p = r05[k]['perplexity_inversion']
    q = r025[k][key]
    print(f"G{k}   | {p['ppl_under_g0']:10.2f}   {p['ppl_under_gk']:10.2f}   {p['perplexity_inversion_ratio']:8.3f}  |  {q['ppl_under_g0']:10.2f}   {q['ppl_under_gk']:10.2f}   {q['perplexity_inversion_ratio']:8.3f}")