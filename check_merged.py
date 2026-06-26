import json
with open("phase3_measurements/results/all_measurements_merged.json") as f:
    merged = json.load(f)
print("Final merged file — all layers check:")
print("  Gens:", [r["generation_k"] for r in merged])
for r in merged:
    k    = r["generation_k"]
    lex  = r.get("lexical") or {}
    sem  = r.get("semantic") or {}
    ppl  = r.get("perplexity_inversion") or {}
    tail = r.get("tail_mass") or {}
    print(
        f"  G{k}:"
        f"  ttr={lex.get('ttr', 'NULL')}"
        f"  cos={sem.get('avg_pairwise_cosine_dist', 'NULL')}"
        f"  ppl_Gk={ppl.get('ppl_under_gk', 'NULL')}"
        f"  tail={tail.get('tail_mass_fraction', 'NULL')}"
    )