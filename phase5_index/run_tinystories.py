"""
MCO Phase 5 — TinyStories Contamination Index Pilot
=====================================================
Applies the MCO measurement framework to TinyStories, a 100% GPT-4 generated
dataset. Known ground truth: should produce high contamination index score.

This is the validation test for the measurement framework.

Usage:
    python phase5_index/run_tinystories.py

Requires: datasets, sentence-transformers, torch, scipy
Runtime: ~2 hours on CPU, ~30 min with GPU
"""

import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch


REFERENCE_PACK = Path("phase1_baseline/reference_pack.pkl")
OUTPUT_DIR     = Path("phase5_index/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES      = 1000   # sample size for pilot — manageable on CPU
SEED           = 42
DATASET_NAME   = "roneneldan/TinyStories"

CALIBRATION = {
    "ttr_g0": 0.112, "ttr_g3": 0.058,
    "kl_g0": 5.05662, "kl_g3": 5.38296,
}
SIGNAL_WEIGHTS = {
    "lexical_ttr":        0.865,
    "lexical_kl":         0.865,
    "ppl_predictability": 0.865,
    "semantic_coverage":  0.300,
    # tail mass excluded — domain mismatch makes it unreliable, same as run_dataset.py
}
WEIGHT_SUM = sum(SIGNAL_WEIGHTS.values())


def load_tinystories(n=N_SAMPLES, seed=SEED):
    print(f"Loading TinyStories (n={n})...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    docs = []
    for item in ds:
        text = item.get("text", item.get("story", ""))
        if text and len(text.split()) >= 20:
            docs.append(text.strip())
        if len(docs) >= n:
            break
    print(f"  Loaded {len(docs)} TinyStories documents")
    print(f"  Sample: {docs[0][:120]}...")
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

    def ngram_entropy(n):
        counts = Counter()
        for toks in tokenized:
            for i in range(len(toks)-n+1):
                counts[tuple(toks[i:i+n])] += 1
        s = sum(counts.values())
        return -sum((c/s)*math.log2(c/s) for c in counts.values()) if s else 0

    kl = 0.0
    if uni_base and total > 0:
        bt = sum(uni_base.values())
        vs = len(uni_base)
        for w, c in vocab.items():
            gp = c / total
            raw = uni_base.get(w, 0.0)
            b = (raw * bt + lp) / (bt + lp * (vs + 1))
            if gp > 0 and b > 0:
                kl += gp * math.log2(gp / b)
        kl = max(0.0, kl)

    sorted_counts = sorted(vocab.values(), reverse=True)
    zipf_alpha = None
    if len(sorted_counts) >= 10:
        ranks = np.arange(1, len(sorted_counts)+1, dtype=float)
        freqs = np.array(sorted_counts, dtype=float)
        A = np.column_stack([np.ones_like(ranks), np.log(ranks)])
        res = np.linalg.lstsq(A, np.log(freqs+1e-9), rcond=None)
        zipf_alpha = float(-res[0][1])

    lex_b = pack.get("lexical", {})
    be = lex_b.get("entropy_1gram", None)
    e1g = ngram_entropy(1)

    return {
        "ttr": round(ttr, 6),
        "entropy_1gram": round(e1g, 6),
        "entropy_3gram": round(ngram_entropy(3), 6),
        "kl_div_1gram": round(kl, 6),
        "zipf_alpha": round(zipf_alpha, 6) if zipf_alpha else None,
        "baseline_entropy_1gram": be,
        "entropy_rel_change": round((e1g-be)/be, 6) if be else None,
    }


def measure_semantic(docs, pack, encoder):
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors

    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]
    sem_b = pack.get("semantic", {})

    print("  Embedding TinyStories documents...")
    gen_emb = encoder.encode(
        docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    gen_pca = pca.transform(gen_emb)

    # Cosine distance
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(gen_emb), size=min(500, len(gen_emb)), replace=False)
    sub = gen_emb[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms==0, 1, norms)
    normed = sub / norms
    sim = normed @ normed.T
    upper = np.triu_indices(len(normed), k=1)
    cos_dist = float(np.mean(1 - sim[upper]))

    # NN coverage (fixed metric from Fix 3)
    nbrs = NearestNeighbors(n_neighbors=2).fit(baseline_pca)
    baseline_self_dists, _ = nbrs.kneighbors(baseline_pca)
    threshold = float(np.percentile(baseline_self_dists[:, 1], 90))
    gen_dists, _ = nbrs.kneighbors(gen_pca)
    coverage = float(np.mean(gen_dists[:, 0] <= threshold))

    bc = sem_b.get("avg_pairwise_cosine_distance",
         sem_b.get("avg_cosine_distance", None))

    return {
        "avg_pairwise_cosine_dist": round(cos_dist, 6),
        "semantic_coverage": round(coverage, 6),
        "baseline_avg_cosine_dist": bc,
        "cosine_rel_change": round((cos_dist-bc)/bc, 6) if bc else None,
    }


def measure_tail_mass(docs, pack, encoder):
    kde = pack["kde"]
    threshold = float(pack["tail_threshold"])
    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]

    gen_emb = encoder.encode(
        docs, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    gen_pca = pca.transform(gen_emb)

    gen_ll = kde.score_samples(gen_pca)
    base_ll = kde.score_samples(baseline_pca)
    tail_frac = float(np.mean(gen_ll < threshold))
    base_tail = float(np.mean(base_ll < threshold))

    return {
        "mean_log_likelihood": round(float(np.mean(gen_ll)), 6),
        "std_log_likelihood": round(float(np.std(gen_ll)), 6),
        "tail_mass_fraction": round(tail_frac, 6),
        "tail_threshold": round(threshold, 6),
        "baseline_tail_fraction": round(base_tail, 6),
        "tail_fraction_rel_change": round((tail_frac-base_tail)/base_tail, 6)
            if base_tail > 0 else None,
    }


def measure_ppl_inversion(docs, pack, device="cpu", max_s=200):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(SEED)
    samples = [docs[i] for i in sorted(
        rng.choice(len(docs), size=min(max_s, len(docs)), replace=False).tolist())]

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("  Loading G0 reference model...")
    g0 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    g0.eval()
    for p in g0.parameters(): p.requires_grad = False

    # For Phase 5: Gk IS G0 — we're measuring how synthetic the text is
    # PPL ratio for contamination index: PPL_G0(text) vs expected PPL
    # For 100% synthetic text (TinyStories), G0 should find it more predictable
    # because it was generated by a GPT-class model similar to G0's training
    # We measure PPL_G0 distribution vs Phase 1 Wikipedia baseline PPL

    print(f"  Scoring {len(samples)} TinyStories samples under G0...")
    ppls = []
    with torch.no_grad():
        for text in samples:
            enc = tok(text[:512], return_tensors="pt", truncation=True,
                      max_length=128).to(device)
            if enc.input_ids.size(1) < 2: continue
            out = g0(**enc, labels=enc.input_ids.clone())
            ppl = math.exp(out.loss.item())
            if math.isfinite(ppl): ppls.append(ppl)

    ppl_baseline_mean = pack.get("ppl_baseline", {}).get("mean_ppl", 44.44)

    mean_ppl = float(np.mean(ppls))
    # Ratio: how much lower is synthetic PPL vs human baseline PPL?
    # Lower synthetic PPL = model finds it more predictable = more synthetic-like
    ppl_ratio = ppl_baseline_mean / mean_ppl if mean_ppl > 0 else None

    return {
        "mean_ppl_under_g0": round(mean_ppl, 4),
        "std_ppl_under_g0": round(float(np.std(ppls)), 4),
        "baseline_mean_ppl": ppl_baseline_mean,
        "ppl_ratio_vs_baseline": round(ppl_ratio, 6) if ppl_ratio else None,
        "n_samples": len(ppls),
        "ppl_per_sample": [round(p, 4) for p in ppls],
    }


def compute_composite_index(lex, sem, tail, ppl):
    """
    Composite contamination index: 0=clean, 1=fully synthetic.

    CANONICAL FORMULA — identical to phase5_index/run_dataset.py's
    compute_composite(). Unified to fix a formula divergence that made
    TinyStories not directly comparable to C4/Pile-CC/Wikipedia holdout
    scores in the same table. `tail` param kept for signature
    compatibility with callers; unused, consistent with run_dataset.py.
    """
    c = CALIBRATION
    component_scores = {}

    ttr = lex.get("ttr", c["ttr_g0"])
    ttr_score = max(0.0, min(1.0, (c["ttr_g0"] - ttr) / (c["ttr_g0"] - c["ttr_g3"])))
    component_scores["lexical_ttr"] = round(ttr_score, 4)

    kl = lex.get("kl_div_1gram", c["kl_g0"])
    kl_score = max(0.0, min(1.0, (kl - c["kl_g0"]) / (c["kl_g3"] - c["kl_g0"])))
    component_scores["lexical_kl"] = round(kl_score, 4)

    ppl_r = ppl.get("ppl_ratio_vs_baseline") or 1.0
    ppl_score = max(0.0, min(1.0, (ppl_r - 1.0) / (2.73 - 1.0)))
    component_scores["ppl_predictability"] = round(ppl_score, 4)

    cov = sem.get("semantic_coverage", 1.0)
    cov_score = max(0.0, min(1.0, 1.0 - cov))
    component_scores["semantic_coverage"] = round(cov_score, 4)

    weighted_sum = sum(SIGNAL_WEIGHTS[k] * component_scores[k] for k in SIGNAL_WEIGHTS)
    composite = round(weighted_sum / WEIGHT_SUM, 4)

    return composite, component_scores


def main():
    print("=" * 60)
    print("MCO Phase 5 — TinyStories Contamination Index")
    print("=" * 60)

    # Load reference pack
    print(f"\nLoading reference pack...")
    with open(REFERENCE_PACK, "rb") as f:
        pack = pickle.load(f)

    # Load supplementary files
    for fname, key in [
        ("phase1_baseline/measurements/lexical_baseline.json", "lexical"),
        ("phase1_baseline/measurements/semantic_baseline.json", "semantic"),
        ("phase1_baseline/measurements/ppl_baseline.json", "ppl_baseline"),
        ("phase1_baseline/measurements/kl_baseline_distributions.json", "kl_distributions"),
    ]:
        with open(fname) as f:
            pack[key] = json.load(f)
    print("  Reference pack + baselines loaded.")

    # Load encoder
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    encoder = SentenceTransformer(pack.get("encoder_id",
        "sentence-transformers/all-MiniLM-L6-v2"))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Encoder loaded and frozen.")

    # Load TinyStories
    docs = load_tinystories(n=N_SAMPLES)

    # Run all four layers
    t_start = time.time()

    print("\n[1/4] Lexical...")
    lex = measure_lexical(docs, pack)
    print(f"  ttr={lex['ttr']:.4f}  entropy={lex['entropy_1gram']:.4f}  kl={lex['kl_div_1gram']:.4f}")

    print("\n[2/4] Semantic...")
    sem = measure_semantic(docs, pack, encoder)
    print(f"  cos_dist={sem['avg_pairwise_cosine_dist']:.4f}  coverage={sem['semantic_coverage']:.4f}")

    print("\n[3/4] Tail mass (Phase 1 Wikipedia reference)...")
    tail = measure_tail_mass(docs, pack, encoder)
    print(f"  tail_fraction={tail['tail_mass_fraction']:.5f}  (baseline={tail['baseline_tail_fraction']:.5f})")

    print("\n[4/4] PPL under G0...")
    ppl = measure_ppl_inversion(docs, pack, device=device)
    print(f"  mean_ppl={ppl['mean_ppl_under_g0']:.2f}  ppl_ratio_vs_baseline={ppl['ppl_ratio_vs_baseline']}")

    # Composite index
    composite, component_scores = compute_composite_index(lex, sem, tail, ppl)
    elapsed = round((time.time() - t_start) / 60, 1)

    # Build index entry
    entry = {
        "dataset_name": "roneneldan/TinyStories",
        "dataset_version": "1.0",
        "dataset_source_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "ground_truth": "100% GPT-4 generated (known synthetic)",
        "sample_size": len(docs),
        "sample_strategy": "random streaming, seed=42",
        "measurement_date": datetime.now().strftime("%Y-%m-%d"),
        "encoder_id": pack.get("encoder_id"),
        "reference_model_id": "distilgpt2",
        "elapsed_minutes": elapsed,
        "scores": {
            "lexical_entropy_kl": lex["kl_div_1gram"],
            "semantic_coverage": sem["semantic_coverage"],
            "tail_mass_fraction": tail["tail_mass_fraction"],
            "ppl_ratio_vs_baseline": ppl["ppl_ratio_vs_baseline"],
            "composite_contamination_index": composite,
        },
        "component_scores": component_scores,
        "estimated_synthetic_fraction": {
            "point_estimate": composite,
            "ci_lower_95": max(0.0, composite - 0.15),
            "ci_upper_95": min(1.0, composite + 0.15),
            "estimation_method": "calibrated_against_phase3_distilgpt2_r05",
        },
        "full_measurements": {
            "lexical": lex,
            "semantic": sem,
            "tail_mass": tail,
            "ppl": ppl,
        },
        "caveats": [
            "Calibrated against DistilGPT-2 simulations at 5k-document scale, R=0.5",
            "Encoder: all-MiniLM-L6-v2. Results not comparable across encoder choices.",
            "Sample size 1k documents; pilot scale only",
            "Tail mass reference is Phase 1 Wikipedia KDE — domain mismatch possible",
            "PPL ratio uses G0 (DistilGPT-2) as reference — scale-dependent",
        ],
    }

    # Save
    out_path = OUTPUT_DIR / "tinystories_index.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 5 RESULTS — TinyStories Contamination Index")
    print("=" * 60)
    print(f"  Dataset: roneneldan/TinyStories (100% GPT-4 synthetic)")
    print(f"  Sample: {len(docs)} documents")
    print()
    print(f"  COMPOSITE CONTAMINATION INDEX: {composite:.4f} / 1.0")
    print()
    print("  Component scores:")
    for k, v in component_scores.items():
        bar = "█" * int(v * 20)
        print(f"    {k:<25} {v:.4f}  |{bar:<20}|")
    print()
    print("  Raw measurements:")
    print(f"    TTR:           {lex['ttr']:.4f}  (Wikipedia baseline: 0.1423)")
    print(f"    Entropy 1gram: {lex['entropy_1gram']:.4f}  (baseline: 11.93)")
    print(f"    KL divergence: {lex['kl_div_1gram']:.4f}  (G0: 5.06, G3: 5.38)")
    print(f"    Semantic cov:  {sem['semantic_coverage']:.4f}")
    print(f"    Tail fraction: {tail['tail_mass_fraction']:.5f}  (baseline: {tail['baseline_tail_fraction']:.5f})")
    print(f"    PPL under G0:  {ppl['mean_ppl_under_g0']:.2f}  (Wikipedia baseline: 44.44)")
    print(f"    PPL ratio:     {ppl['ppl_ratio_vs_baseline']:.4f}")
    print()
    print(f"  Written to: {out_path}")
    print(f"  Runtime: {elapsed} min")
    print()

    # Validation check
    if composite > 0.5:
        print("  ✓ VALIDATION PASSED: composite index > 0.5 for known synthetic dataset")
        print("  The measurement framework correctly identifies TinyStories as highly")
        print("  contaminated with synthetic content.")
    else:
        print("  ✗ VALIDATION CONCERN: composite index < 0.5 for known synthetic dataset")
        print("  Either the calibration needs adjustment or TinyStories has different")
        print("  statistical properties from DistilGPT-2 outputs at 135-token scale.")
        print("  Investigate before publishing Phase 5 results.")


if __name__ == "__main__":
    main()