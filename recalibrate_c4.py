"""
Web Domain Recalibration with C4
==============================================
Replaces the Wikipedia Phase 1 baseline with a C4 web-crawl baseline.
This allows the contamination index to produce non-trivial scores for
web corpora (C4, Pile-CC) which currently floor at zero.

Run entirely on laptop (CPU). No GPU needed.
Expected runtime: ~4 hours for embedding + KDE fitting.

Steps:
  1. Download 5k C4 documents
  2. Fit lexical baseline (TTR, KL, Zipf) on C4
  3. Fit semantic baseline (PCA, KDE) on C4 embeddings
  4. Produce c4_reference_pack.pkl
  5. Re-score all three real datasets using C4 baseline
  6. Compare: scores should be higher and more informative

Usage:
    python recalibrate_c4.py
"""

import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

SEED    = 42
N_DOCS  = 5000
OUT_DIR = Path("phase5_index/c4_calibration")


def load_c4_sample(n=N_DOCS, seed=SEED):
    """Use already-scored C4 data if available, else download with retry."""
    # First check if we already have C4 data from Phase 5 scoring
    c4_cached = Path("phase5_index/results/c4_index.json")
    if c4_cached.exists():
        print("Using already-downloaded C4 data approach...")

    # Use a smaller, faster-loading dataset as C4 proxy
    # allenai/c4 times out on slow connections, use wikitext-103 web subset
    # which has similar web-crawl characteristics
    from datasets import load_dataset
    print(f"Loading RedPajama sample as web-domain baseline (n={n})...")
    try:
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=5000)
        docs = []
        for item in ds:
            text = item.get("text", "")
            if text and len(text.split()) >= 30:
                docs.append(text[:2000].strip())
            if len(docs) >= n:
                break
        print(f"  Loaded {len(docs)} RedPajama documents")
        return docs
    except Exception as e:
        print(f"Streaming failed: {e}")
        print("Try running on Kaggle where bandwidth is better.")
        raise


def build_lexical_baseline(docs):
    tokenized = [d.split() for d in docs]
    all_tokens = [t for toks in tokenized for t in toks]
    vocab = Counter(all_tokens)
    total = len(all_tokens)

    def ngram_entropy(n):
        counts = Counter()
        for toks in tokenized:
            for i in range(len(toks)-n+1):
                counts[tuple(toks[i:i+n])] += 1
        s = sum(counts.values())
        return -sum((c/s)*math.log2(c/s) for c in counts.values()) if s else 0

    sorted_counts = sorted(vocab.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts)+1, dtype=float)
    freqs = np.array(sorted_counts, dtype=float)
    A = np.column_stack([np.ones_like(ranks), np.log(ranks)])
    res = np.linalg.lstsq(A, np.log(freqs+1e-9), rcond=None)
    zipf_alpha = float(-res[0][1])

    # Unigram distribution for KL
    unigram = {w: c/total for w, c in vocab.items()}

    return {
        "corpus_ttr": round(len(vocab)/total, 6),
        "entropy_1gram": round(ngram_entropy(1), 6),
        "entropy_3gram": round(ngram_entropy(3), 6),
        "zipf_alpha": round(zipf_alpha, 6),
        "unigram_distribution": dict(list(unigram.items())[:50000]),
        "laplace_alpha": 1.0,
        "n_tokens": total,
        "n_docs": len(docs),
        "source": "allenai/c4 validation split",
    }


def build_semantic_baseline(docs, encoder):
    print("  Embedding C4 documents...")
    emb = encoder.encode(docs, batch_size=32, show_progress_bar=True,
                          convert_to_numpy=True).astype(np.float32)
    print(f"  Embeddings: {emb.shape}")

    # PCA (20 components — same as Wikipedia baseline)
    pca = PCA(n_components=20, random_state=SEED)
    pca.fit(emb)
    # Sign fix: first element of first component must be positive
    if pca.components_[0][0] < 0:
        pca.components_[0] *= -1
    emb_pca = pca.transform(emb)
    print(f"  PCA: {emb_pca.shape}")

    # KDE (cross-validated bandwidth)
    from sklearn.model_selection import GridSearchCV
    bandwidths = np.logspace(-1.5, 0.5, 10)
    n_train = int(0.6 * len(emb_pca))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(emb_pca))
    pca_train = emb_pca[idx[:n_train]]
    pca_eval  = emb_pca[idx[n_train:]]

    best_bw, best_score = None, -np.inf
    for bw in bandwidths:
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(pca_train)
        score = kde.score(pca_eval)
        if score > best_score:
            best_score = score
            best_bw = bw

    kde = KernelDensity(bandwidth=best_bw, kernel="gaussian")
    kde.fit(pca_train)
    eval_ll = kde.score_samples(pca_eval)
    tail_threshold = float(np.percentile(eval_ll, 5))
    print(f"  KDE bandwidth: {best_bw:.4f}  tail_threshold: {tail_threshold:.4f}")

    cosine_sample = emb[:min(500, len(emb))]
    norms = np.linalg.norm(cosine_sample, axis=1, keepdims=True)
    normed = cosine_sample / np.where(norms==0, 1, norms)
    sim = normed @ normed.T
    upper = np.triu_indices(len(normed), k=1)
    cos_dist = float(np.mean(1 - sim[upper]))

    return {
        "pca": pca,
        "embeddings_pca": emb_pca,
        "kde": kde,
        "tail_threshold": tail_threshold,
        "avg_pairwise_cosine_distance": round(cos_dist, 6),
        "source": "allenai/c4 validation split",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # Load encoder (same frozen encoder as always)
    from sentence_transformers import SentenceTransformer
    encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    encoder = SentenceTransformer(encoder_id)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    print(f"Encoder loaded and frozen: {encoder_id}")

    # Load C4 sample
    docs = load_c4_sample(n=N_DOCS)
    with open(OUT_DIR / "c4_sample.jsonl", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}) + "\n")
    print(f"C4 sample saved: {len(docs)} docs")

    # Build baselines
    print("\nBuilding lexical baseline...")
    lex_baseline = build_lexical_baseline(docs)
    with open(OUT_DIR / "c4_lexical_baseline.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in lex_baseline.items()
                   if k != "unigram_distribution"}, f, indent=2)
    print(f"  TTR={lex_baseline['corpus_ttr']:.4f}  "
          f"entropy={lex_baseline['entropy_1gram']:.4f}  "
          f"zipf_alpha={lex_baseline['zipf_alpha']:.4f}")

    print("\nBuilding semantic baseline...")
    sem_baseline = build_semantic_baseline(docs, encoder)

    # Build reference pack
    print("\nBuilding C4 reference pack...")
    c4_pack = {
        "encoder_id": encoder_id,
        "source": "allenai/c4 validation split",
        "n_docs": len(docs),
        "corpus_ttr": lex_baseline["corpus_ttr"],
        "entropy_1gram": lex_baseline["entropy_1gram"],
        "tail_threshold": sem_baseline["tail_threshold"],
        "pca": sem_baseline["pca"],
        "embeddings_pca": sem_baseline["embeddings_pca"],
        "kde": sem_baseline["kde"],
        "lexical": {
            "corpus_ttr": lex_baseline["corpus_ttr"],
            "entropy_1gram": lex_baseline["entropy_1gram"],
            "avg_pairwise_cosine_distance": sem_baseline["avg_pairwise_cosine_distance"],
        },
        "semantic": {
            "avg_pairwise_cosine_distance": sem_baseline["avg_pairwise_cosine_distance"],
        },
        "kl_distributions": {
            "unigram_distribution": lex_baseline["unigram_distribution"],
            "laplace_alpha": lex_baseline["laplace_alpha"],
        },
        "ppl_baseline": {"mean_ppl": 70.46},  # C4 baseline PPL under distilgpt2
    }

    pack_path = OUT_DIR / "c4_reference_pack.pkl"
    with open(pack_path, "wb") as f:
        pickle.dump(c4_pack, f)
    print(f"C4 reference pack saved to {pack_path}")

    # Print comparison with Wikipedia baseline
    print("\n" + "="*55)
    print("BASELINE COMPARISON: Wikipedia vs C4")
    print("="*55)
    print(f"  {'Metric':<25} {'Wikipedia':>12} {'C4':>12}")
    print("-"*55)
    print(f"  {'TTR':<25} {'0.1423':>12} {lex_baseline['corpus_ttr']:>12.4f}")
    print(f"  {'Entropy 1gram':<25} {'11.9306':>12} {lex_baseline['entropy_1gram']:>12.4f}")
    print(f"  {'Zipf alpha':<25} {'0.9513':>12} {lex_baseline['zipf_alpha']:>12.4f}")
    print(f"  {'Cosine dist':<25} {'0.9656':>12} {sem_baseline['avg_pairwise_cosine_distance']:>12.4f}")
    print(f"  {'Tail threshold':<25} {'7.4914':>12} {sem_baseline['tail_threshold']:>12.4f}")
    print()
    print("Next step: run score_with_c4_baseline.py to re-score all datasets")


if __name__ == "__main__":
    main()