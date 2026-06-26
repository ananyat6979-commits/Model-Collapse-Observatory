import json

with open("phase3_measurements/results/all_measurements_merged.json") as f:
    merged = json.load(f)

with open("phase3_measurements/results/measurements_G4_complete.json") as f:
    g4 = json.load(f)

# Load G4 PPL and tail from measurements_v2_G4.json
with open("phase3_measurements/results/measurements_v2_G4.json") as f:
    g4_v2 = json.load(f)

# Inject PPL and tail into g4 complete
g4["perplexity_inversion"] = g4_v2["perplexity_inversion_fixed"]
g4["tail_mass"] = g4_v2["tail_mass_fixed"]

# Replace G4 entry in merged file
merged = [r for r in merged if r["generation_k"] != 4]
merged.append(g4)
merged.sort(key=lambda x: x["generation_k"])

with open("phase3_measurements/results/all_measurements_merged.json", "w") as f:
    json.dump(merged, f, indent=2)

print("Merged. Generations:", [r["generation_k"] for r in merged])
for r in merged:
    k   = r["generation_k"]
    lex = r.get("lexical") or {}
    ppl = r.get("perplexity_inversion") or {}
    ttr = lex.get("ttr", "null")
    gk  = ppl.get("ppl_under_gk", "null")
    print(f"  G{k}: ttr={ttr}  ppl_Gk={gk}")