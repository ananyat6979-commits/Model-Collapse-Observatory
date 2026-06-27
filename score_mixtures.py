"""
Score Calibration Mixtures
=========================================
Applies the MCO measurement framework to each calibration mixture
and produces the ground-truth calibration curve.

Usage:
    python score_mixtures.py

Output:
    phase5_index/calibration/calibration_results.json
    phase5_index/calibration/calibration_curve.txt  (printable table)
"""

import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

REFERENCE_PACK = Path("phase1_baseline/reference_pack.pkl")
CALIBRATION_DIR = Path("phase5_index/calibration")
SEED = 42

SIGNAL_WEIGHTS = {
    "lexical_ttr":        0.865,
    "lexical_kl":         0.865,
    "ppl_predictability": 0.865,
    "semantic_coverage":  0.300,
}
WEIGHT_SUM = sum(SIGNAL_WEIGHTS.values())

CALIBRATION = {
    "ttr_g0": 0.11227, "ttr_g3": 0.05847,
    "kl_g0":  5.05662, "kl_g3":  5.38296,
    "ppl_baseline": 44.44,
}


def load_mixture(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("text", "").strip():
                docs.append(d["text"])
    return docs


def measure_lexical(docs, pack):
    kl_dists = pack.get("kl_distributions", {})
    uni_base = kl_dists.get("unigram_distribution", {})
    lp = float(kl_dists.get("laplace_alpha", 1.0))
    tokenized = [d.split() for d in docs]
    all_tokens = [t for toks in tokenized for t in toks]
    vocab = Counter(all_tokens)
    total = len(all_tokens)
    ttr = len(vocab) / total if total else 0

    kl = 0.0
    if uni_base and total > 0:
        bt = sum(uni_base.values())
        vs = len(uni_base)
        for w, c in vocab.items():
            gp = c / total
            b = (uni_base.get(w, 0.0) * bt + lp) / (bt + lp * (vs + 1))
            if gp > 0 and b > 0:
                kl += gp * math.log2(gp / b)
        kl = max(0.0, kl)

    return {"ttr": round(ttr, 6), "kl_div_1gram": round(kl, 6)}


def measure_semantic(docs, pack, encoder):
    from sklearn.neighbors import NearestNeighbors
    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]
    gen_emb = encoder.encode(docs, batch_size=32, show_progress_bar=False,
                              convert_to_numpy=True).astype(np.float32)
    gen_pca = pca.transform(gen_emb)
    nbrs = NearestNeighbors(n_neighbors=2).fit(baseline_pca)
    base_dists, _ = nbrs.kneighbors(baseline_pca)
    threshold = float(np.percentile(base_dists[:, 1], 90))
    gen_dists, _ = nbrs.kneighbors(gen_pca)
    coverage = float(np.mean(gen_dists[:, 0] <= threshold))
    return {"semantic_coverage": round(coverage, 6)}


def measure_ppl(docs, pack, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)
    n = min(200, len(docs))
    samples = [docs[i] for i in sorted(rng.choice(len(docs), n, replace=False).tolist())]

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    g0 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    g0.eval()
    for p in g0.parameters(): p.requires_grad = False

    ppls = []
    with torch.no_grad():
        for text in samples:
            enc = tok(text[:1024], return_tensors="pt", truncation=True,
                      max_length=128).to(device)
            if enc.input_ids.size(1) < 2: continue
            out = g0(**enc, labels=enc.input_ids.clone())
            ppl = math.exp(out.loss.item())
            if math.isfinite(ppl) and ppl < 10000:
                ppls.append(ppl)
    del g0
    ppl_baseline = pack.get("ppl_baseline", {}).get("mean_ppl", 44.44)
    mean_ppl = float(np.mean(ppls)) if ppls else float("nan")
    ratio = ppl_baseline / mean_ppl if mean_ppl > 0 else None
    return {"mean_ppl_under_g0": round(mean_ppl, 4),
            "ppl_ratio_vs_baseline": round(ratio, 6) if ratio else None}


def compute_composite(lex, sem, ppl_result):
    c = CALIBRATION
    ttr = lex.get("ttr", c["ttr_g0"])
    ttr_score = max(0.0, min(1.0, (c["ttr_g0"] - ttr) / (c["ttr_g0"] - c["ttr_g3"])))

    kl = lex.get("kl_div_1gram", c["kl_g0"])
    kl_score = max(0.0, min(1.0, (kl - c["kl_g0"]) / (c["kl_g3"] - c["kl_g0"])))

    ppl_r = ppl_result.get("ppl_ratio_vs_baseline") or 1.0
    ppl_score = max(0.0, min(1.0, (ppl_r - 1.0) / (2.73 - 1.0)))

    cov = sem.get("semantic_coverage", 1.0)
    cov_score = max(0.0, min(1.0, 1.0 - cov))

    scores = {
        "lexical_ttr":        round(ttr_score, 4),
        "lexical_kl":         round(kl_score, 4),
        "ppl_predictability": round(ppl_score, 4),
        "semantic_coverage":  round(cov_score, 4),
    }
    composite = round(sum(SIGNAL_WEIGHTS[k] * v for k, v in scores.items()) / WEIGHT_SUM, 4)
    return composite, scores


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading reference pack...")
    with open(REFERENCE_PACK, "rb") as f:
        pack = pickle.load(f)
    for fname, key in [
        ("phase1_baseline/measurements/lexical_baseline.json",          "lexical"),
        ("phase1_baseline/measurements/semantic_baseline.json",         "semantic"),
        ("phase1_baseline/measurements/ppl_baseline.json",              "ppl_baseline"),
        ("phase1_baseline/measurements/kl_baseline_distributions.json", "kl_distributions"),
    ]:
        with open(fname, encoding="utf-8") as f:
            pack[key] = json.load(f)

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(pack.get("encoder_id",
        "sentence-transformers/all-MiniLM-L6-v2"))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    print(f"Encoder loaded. Device: {device}")

    # Find all mixture files
    mixture_files = sorted(CALIBRATION_DIR.glob("mixture_*pct_synthetic.jsonl"))
    if not mixture_files:
        print("No mixture files found. Run create_mixtures.py first.")
        return

    all_results = []
    for mpath in mixture_files:
        frac_pct = int(mpath.stem.split("_")[1].replace("pct", ""))
        frac = frac_pct / 100.0
        docs = load_mixture(mpath)
        print(f"\n[{frac_pct:3d}% synthetic] {len(docs)} docs")
        t = time.time()

        lex = measure_lexical(docs, pack)
        sem = measure_semantic(docs, pack, encoder)
        ppl = measure_ppl(docs, pack, device)
        composite, scores = compute_composite(lex, sem, ppl)

        elapsed = round((time.time() - t) / 60, 1)
        print(f"  ttr={lex['ttr']:.4f}  kl={lex['kl_div_1gram']:.4f}  "
              f"ppl_ratio={ppl.get('ppl_ratio_vs_baseline', 'N/A')}  "
              f"composite={composite:.4f}  ({elapsed}min)")

        all_results.append({
            "true_synthetic_fraction": frac,
            "true_synthetic_pct": frac_pct,
            "n_docs": len(docs),
            "composite_index": composite,
            "component_scores": scores,
            "raw": {"lexical": lex, "semantic": sem, "ppl": ppl},
            "elapsed_minutes": elapsed,
        })

    # Save results
    out_path = CALIBRATION_DIR / "calibration_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print calibration table
    print("\n" + "=" * 55)
    print("CALIBRATION CURVE — Composite Index vs True Synthetic Fraction")
    print("=" * 55)
    print(f"  {'True %':>7} | {'Composite':>10} | {'TTR':>7} | {'PPL ratio':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"  {r['true_synthetic_pct']:>6}% | "
              f"{r['composite_index']:>10.4f} | "
              f"{r['raw']['lexical']['ttr']:>7.4f} | "
              f"{str(r['raw']['ppl'].get('ppl_ratio_vs_baseline', 'N/A')):>10}")

    # Test monotonicity
    composites = [r["composite_index"] for r in all_results]
    fractions  = [r["true_synthetic_fraction"] for r in all_results]
    corr, p = stats.spearmanr(fractions, composites)
    print(f"\nSpearman correlation: rho={corr:.4f}  p={p:.6f}")
    if corr > 0.8 and p < 0.05:
        print("HYPOTHESIS SUPPORTED: composite index tracks synthetic fraction monotonically.")
    else:
        print("HYPOTHESIS NOT SUPPORTED: index is not monotonic with synthetic fraction.")
        print("Calibration needs adjustment before publishing contamination index.")

    # Save curve text for paper
    curve_path = CALIBRATION_DIR / "calibration_curve.txt"
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("True Synthetic %  |  Composite Index\n")
        for r in all_results:
            f.write(f"  {r['true_synthetic_pct']:>3}%  |  {r['composite_index']:.4f}\n")
        f.write(f"\nSpearman rho={corr:.4f}  p={p:.6f}\n")
    print(f"Curve written to {curve_path}")


if __name__ == "__main__":
    main()