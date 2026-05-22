"""
MCO Phase 5 — Real Dataset Contamination Index
===============================================
Scores multiple real training datasets using the MCO measurement framework.

Datasets:
  1. Wikipedia held-out (negative control — known human)
  2. C4 validation split (web-crawled, expected low-moderate contamination)
  3. The Pile Common Crawl subset (expected moderate contamination)

Usage:
    python phase5_index/run_dataset.py --dataset wikipedia_holdout
    python phase5_index/run_dataset.py --dataset c4
    python phase5_index/run_dataset.py --dataset pile_cc

Requires: datasets, sentence-transformers, torch, scipy
"""

import argparse
import json
import math
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ── Configuration ─────────────────────────────────────────────────────────────

REFERENCE_PACK   = Path("phase1_baseline/reference_pack.pkl")
CORPUS_FILE      = Path("phase1_baseline/corpus/documents.jsonl")
OUTPUT_DIR       = Path("phase5_index/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED      = 42
N_SAMPLES = 1000   # documents per dataset
PPL_SAMPLES = 200  # PPL subsample

# Effect sizes from Phase 4 Mann-Whitney tests — used for principled weighting
# Higher effect_r = more reliable signal = higher weight in composite index
SIGNAL_WEIGHTS = {
    "lexical_ttr":        0.865,  # effect_r from G0 vs G1 Mann-Whitney
    "lexical_kl":         0.865,
    "ppl_predictability": 0.865,
    "semantic_coverage":  0.300,  # lower weight — unreliable signal at this scale
    # tail mass excluded — domain mismatch makes it unreliable for real datasets
}
WEIGHT_SUM = sum(SIGNAL_WEIGHTS.values())

# Phase 3 calibration anchors (from R=0.5 simulation)
# G0 = clean generated text baseline, G3 = maximally collapsed
CALIBRATION = {
    "ttr_g0":   0.11227,  "ttr_g3":   0.05847,
    "kl_g0":    5.05662,  "kl_g3":    5.38296,
    "ppl_baseline": 44.44,  # Wikipedia human baseline PPL under G0
}


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_wikipedia_holdout(n=N_SAMPLES, seed=SEED):
    """
    20% held-out Wikipedia documents from Phase 1 corpus.
    This is the NEGATIVE CONTROL — known human text.
    Expected composite index: < 0.20
    """
    docs = []
    with open(CORPUS_FILE) as f:
        all_docs = [json.loads(line)["text"] for line in f]

    rng = np.random.default_rng(seed + 999)  # different seed from training split
    n_holdout = int(0.2 * len(all_docs))
    holdout_idx = rng.choice(len(all_docs), size=n_holdout, replace=False)
    holdout = [all_docs[i] for i in holdout_idx]

    sample_idx = rng.choice(len(holdout), size=min(n, len(holdout)), replace=False)
    docs = [holdout[i] for i in sorted(sample_idx)]

    print(f"  Wikipedia holdout: {len(docs)} docs (from {n_holdout} held-out)")
    print(f"  Sample: {docs[0][:120]}...")
    return docs, {
        "dataset_name": "Wikipedia Phase 1 holdout (20%)",
        "ground_truth": "known human — negative control",
        "expected_composite": "< 0.20",
    }


def load_c4(n=N_SAMPLES, seed=SEED):
    """
    C4 validation split — web-crawled, filtered Common Crawl.
    Expected composite index: 0.15-0.35 (moderate web contamination).
    """
    from datasets import load_dataset
    print("  Loading C4 validation split (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True,
                      trust_remote_code=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    docs = []
    for item in ds:
        text = item.get("text", "")
        if text and len(text.split()) >= 30:
            docs.append(text[:2000].strip())  # cap length
        if len(docs) >= n:
            break
    print(f"  C4: {len(docs)} docs loaded")
    print(f"  Sample: {docs[0][:120]}...")
    return docs, {
        "dataset_name": "allenai/c4",
        "dataset_version": "en",
        "dataset_source_url": "https://huggingface.co/datasets/allenai/c4",
        "ground_truth": "unknown — web crawl with quality filtering",
        "expected_composite": "0.15-0.35",
    }


def load_pile_cc(n=N_SAMPLES, seed=SEED):
    """
    The Pile — Common Crawl subset (Pile-CC).
    Most internet-representative pretraining data.
    Expected composite index: 0.20-0.45.
    """
    from datasets import load_dataset
    print("  Loading The Pile Common Crawl subset (streaming)...")
    try:
        ds = load_dataset("EleutherAI/pile", "default", split="train",
                          streaming=True, trust_remote_code=True)
        ds = ds.filter(lambda x: x.get("meta", {}).get("pile_set_name") == "Pile-CC")
    except Exception:
        # Fallback: use the deduplicated version
        print("  Falling back to monology/pile-uncopyrighted...")
        ds = load_dataset("monology/pile-uncopyrighted", split="train",
                          streaming=True, trust_remote_code=True)

    ds = ds.shuffle(seed=seed, buffer_size=10000)
    docs = []
    for item in ds:
        text = item.get("text", "")
        if text and len(text.split()) >= 30:
            docs.append(text[:2000].strip())
        if len(docs) >= n:
            break
    print(f"  Pile-CC: {len(docs)} docs loaded")
    print(f"  Sample: {docs[0][:120]}...")
    return docs, {
        "dataset_name": "EleutherAI/pile (Common Crawl subset)",
        "dataset_source_url": "https://huggingface.co/datasets/EleutherAI/pile",
        "ground_truth": "unknown — internet crawl, expected partial synthetic contamination",
        "expected_composite": "0.20-0.45",
    }


# ── Measurement functions ─────────────────────────────────────────────────────

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

    lex_b = pack.get("lexical", {})
    be = lex_b.get("entropy_1gram", None)
    e1g = ngram_entropy(1)

    return {
        "ttr": round(ttr, 6),
        "entropy_1gram": round(e1g, 6),
        "kl_div_1gram": round(kl, 6),
        "baseline_entropy_1gram": be,
        "entropy_rel_change": round((e1g-be)/be, 6) if be else None,
    }


def measure_semantic(docs, pack, encoder):
    from sklearn.neighbors import NearestNeighbors

    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]
    sem_b = pack.get("semantic", {})

    gen_emb = encoder.encode(
        docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    gen_pca = pca.transform(gen_emb)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(gen_emb), size=min(500, len(gen_emb)), replace=False)
    sub = gen_emb[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms==0, 1, norms)
    normed = sub / norms
    sim = normed @ normed.T
    upper = np.triu_indices(len(normed), k=1)
    cos_dist = float(np.mean(1 - sim[upper]))

    # NN coverage (Fix 3 — scale-robust)
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
    }


def measure_ppl(docs, pack, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(SEED)
    samples = [docs[i] for i in sorted(
        rng.choice(len(docs), size=min(PPL_SAMPLES, len(docs)), replace=False).tolist())]

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ppl_baseline = pack.get("ppl_baseline", {}).get("mean_ppl", 44.44)
    mean_ppl = float(np.mean(ppls)) if ppls else float("nan")
    ppl_ratio = ppl_baseline / mean_ppl if mean_ppl > 0 else None

    return {
        "mean_ppl_under_g0": round(mean_ppl, 4),
        "std_ppl_under_g0": round(float(np.std(ppls)), 4) if ppls else None,
        "baseline_mean_ppl": ppl_baseline,
        "ppl_ratio_vs_baseline": round(ppl_ratio, 6) if ppl_ratio else None,
        "n_samples": len(ppls),
        "ppl_per_sample": [round(p, 4) for p in ppls],
    }


# ── Composite index ───────────────────────────────────────────────────────────

def compute_composite(lex, sem, ppl_result):
    """
    Weighted composite contamination index.

    Weights derived from Phase 4 Mann-Whitney effect sizes.
    Range: 0 (clean human text) to 1 (maximally collapsed synthetic).

    Tail mass excluded — domain-dependent, unreliable for cross-domain application.
    """
    c = CALIBRATION
    component_scores = {}

    # TTR: lower = more contaminated
    # Anchors: G0=0.112 (clean generated), G3=0.058 (collapsed)
    ttr = lex.get("ttr", 0.112)
    ttr_score = max(0.0, min(1.0, (c["ttr_g0"] - ttr) / (c["ttr_g0"] - c["ttr_g3"])))
    component_scores["lexical_ttr"] = round(ttr_score, 4)

    # KL divergence: higher = more contaminated
    kl = lex.get("kl_div_1gram", c["kl_g0"])
    kl_score = max(0.0, min(1.0, (kl - c["kl_g0"]) / (c["kl_g3"] - c["kl_g0"])))
    component_scores["lexical_kl"] = round(kl_score, 4)

    # PPL ratio: higher = G0 finds it more predictable = more synthetic-like
    ppl_r = ppl_result.get("ppl_ratio_vs_baseline") or 1.0
    # Anchors: 1.0 = human baseline PPL, 2.73 = G3 collapsed ratio
    ppl_score = max(0.0, min(1.0, (ppl_r - 1.0) / (2.73 - 1.0)))
    component_scores["ppl_predictability"] = round(ppl_score, 4)

    # Semantic coverage: lower coverage = less diverse = more contaminated
    cov = sem.get("semantic_coverage", 1.0)
    # Inverted: high coverage (diverse) → low contamination score
    cov_score = max(0.0, min(1.0, 1.0 - cov))
    component_scores["semantic_coverage"] = round(cov_score, 4)

    # Weighted composite using effect sizes as weights
    weighted_sum = sum(
        SIGNAL_WEIGHTS[k] * component_scores[k]
        for k in SIGNAL_WEIGHTS
    )
    composite = round(weighted_sum / WEIGHT_SUM, 4)

    return composite, component_scores


# ── Synthetic fraction estimation ─────────────────────────────────────────────

def estimate_synthetic_fraction(composite):
    """
    Point estimate + 95% CI for synthetic fraction.

    Method: linear interpolation on Phase 3 calibration curve.
    G0 (k=0, R=0.5) composite ≈ 0.0 (by construction)
    G1 composite ≈ 0.35 → ~50% synthetic exposure
    G3 composite ≈ 1.0 → ~94% cumulative synthetic exposure

    This is a rough estimate. The true relationship is nonlinear.
    CI width reflects calibration uncertainty, not sampling error.
    """
    # Conservative linear estimate
    point = round(min(1.0, max(0.0, composite)), 3)
    ci_width = 0.20  # ±20% reflects calibration uncertainty
    return {
        "point_estimate": point,
        "ci_lower_95": round(max(0.0, point - ci_width), 3),
        "ci_upper_95": round(min(1.0, point + ci_width), 3),
        "estimation_method": "calibrated_linear_interpolation_phase3_distilgpt2_r05",
        "note": "Rough estimate. True synthetic fraction is unknown for real datasets. "
                "Point estimate is the composite index value itself, not a calibrated probability.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dataset_name: str):
    print("=" * 60)
    print(f"MCO Phase 5 — {dataset_name}")
    print("=" * 60)

    # Load reference pack
    with open(REFERENCE_PACK, "rb") as f:
        pack = pickle.load(f)
    for fname, key in [
        ("phase1_baseline/measurements/lexical_baseline.json",          "lexical"),
        ("phase1_baseline/measurements/semantic_baseline.json",         "semantic"),
        ("phase1_baseline/measurements/ppl_baseline.json",              "ppl_baseline"),
        ("phase1_baseline/measurements/kl_baseline_distributions.json", "kl_distributions"),
    ]:
        with open(fname) as f:
            pack[key] = json.load(f)

    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(pack.get("encoder_id",
        "sentence-transformers/all-MiniLM-L6-v2"))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # Load dataset
    loaders = {
        "wikipedia_holdout": load_wikipedia_holdout,
        "c4":                load_c4,
        "pile_cc":           load_pile_cc,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(loaders)}")

    docs, metadata = loaders[dataset_name]()

    # Measure
    t0 = time.time()

    print("\n[1/3] Lexical...")
    lex = measure_lexical(docs, pack)
    print(f"  ttr={lex['ttr']:.4f}  entropy={lex['entropy_1gram']:.4f}  kl={lex['kl_div_1gram']:.4f}")

    print("\n[2/3] Semantic...")
    sem = measure_semantic(docs, pack, encoder)
    print(f"  cos_dist={sem['avg_pairwise_cosine_dist']:.4f}  coverage={sem['semantic_coverage']:.4f}")

    print("\n[3/3] PPL under G0...")
    ppl = measure_ppl(docs, pack, device=device)
    print(f"  mean_ppl={ppl['mean_ppl_under_g0']:.2f}  ratio={ppl['ppl_ratio_vs_baseline']:.4f}")

    elapsed = round((time.time() - t0) / 60, 1)

    # Composite
    composite, component_scores = compute_composite(lex, sem, ppl)
    synthetic_est = estimate_synthetic_fraction(composite)

    # Build index entry
    entry = {
        **metadata,
        "sample_size": len(docs),
        "sample_strategy": f"random streaming, seed={SEED}",
        "measurement_date": datetime.now().strftime("%Y-%m-%d"),
        "encoder_id": pack.get("encoder_id"),
        "reference_model": "distilgpt2 (82M, pretrained)",
        "elapsed_minutes": elapsed,
        "scores": {
            "ttr":                  lex["ttr"],
            "entropy_1gram":        lex["entropy_1gram"],
            "kl_div_1gram":         lex["kl_div_1gram"],
            "avg_pairwise_cos_dist":sem["avg_pairwise_cosine_dist"],
            "semantic_coverage":    sem["semantic_coverage"],
            "mean_ppl_under_g0":    ppl["mean_ppl_under_g0"],
            "ppl_ratio_vs_baseline":ppl["ppl_ratio_vs_baseline"],
            "composite_contamination_index": composite,
        },
        "component_scores": component_scores,
        "signal_weights_used": SIGNAL_WEIGHTS,
        "estimated_synthetic_fraction": synthetic_est,
        "calibration_anchors": CALIBRATION,
        "caveats": [
            "Calibrated against DistilGPT-2 (82M) simulations at 5k-document scale, R=0.5",
            "Reference encoder: all-MiniLM-L6-v2. Results not comparable across encoder choices",
            "Tail mass excluded from composite index — domain-dependent signal",
            "Semantic coverage: unreliable for diverse synthetic datasets (low weight applied)",
            "PPL ratio assumes G0=distilgpt2. Different reference models will give different scores",
            f"Sample size: {N_SAMPLES} documents — pilot scale only",
            "'Composite index' is not equivalent to 'fraction of synthetic documents'",
        ],
        "full_measurements": {"lexical": lex, "semantic": sem, "ppl": ppl},
    }

    # Save
    out_file = OUTPUT_DIR / f"{dataset_name}_index.json"
    with open(out_file, "w") as f:
        json.dump(entry, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS — {metadata['dataset_name']}")
    print(f"{'='*60}")
    print(f"  Ground truth:  {metadata.get('ground_truth', 'unknown')}")
    print(f"  Expected range:{metadata.get('expected_composite', 'N/A')}")
    print()
    print(f"  COMPOSITE INDEX: {composite:.4f} / 1.0")
    print()
    for sig, score in component_scores.items():
        w = SIGNAL_WEIGHTS.get(sig, 0)
        bar = "█" * int(score * 20)
        print(f"  {sig:<25} {score:.4f}  (weight={w:.3f})  |{bar:<20}|")
    print()
    print(f"  Raw values:")
    print(f"    TTR:         {lex['ttr']:.4f}  (Wikipedia: 0.1423, G3: 0.0585)")
    print(f"    KL div:      {lex['kl_div_1gram']:.4f}  (G0: 5.057, G3: 5.383)")
    print(f"    PPL under G0:{ppl['mean_ppl_under_g0']:.2f}  (Wikipedia: 44.44)")
    print(f"    PPL ratio:   {ppl['ppl_ratio_vs_baseline']:.4f}  (G0: 1.0, G3: ~2.73)")
    print(f"    Semantic cov:{sem['semantic_coverage']:.4f}")
    print()
    print(f"  Written to: {out_file}")
    print(f"  Runtime: {elapsed} min")

    # Validation note for negative control
    if dataset_name == "wikipedia_holdout":
        if composite < 0.25:
            print(f"\n  ✓ NEGATIVE CONTROL PASSED: composite={composite:.4f} < 0.25")
            print("  Framework correctly scores known-human text as low-contamination.")
        else:
            print(f"\n  ✗ NEGATIVE CONTROL FAILED: composite={composite:.4f} >= 0.25")
            print("  Framework over-flags human text. Calibration needs adjustment.")
            print("  Do not publish contamination index until this is resolved.")

    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["wikipedia_holdout", "c4", "pile_cc"])
    args = parser.parse_args()
    run(args.dataset)
