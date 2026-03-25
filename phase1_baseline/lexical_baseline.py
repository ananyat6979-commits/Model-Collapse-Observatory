"""
MCO Phase 1 — Lexical Baseline
================================
Computes and serializes all lexical baseline statistics from the human corpus.

Produces:
    measurements/lexical_baseline.json  — TTR, entropy, vocabulary stats
    measurements/zipf_params.json       — Zipf alpha, R², fit diagnostics

These files are loaded by phase3_measurements/layers/lexical.py to compute
KL divergence of generated text against the human baseline distribution.

Self-test (no corpus needed):
    python phase1_baseline/lexical_baseline.py --self-test
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from scipy.special import kl_div


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def whitespace_tokenize(text: str) -> list[str]:
    """
    Same tokenizer used in ingest.py — must be identical across all phases.
    Lexical metrics are defined on whitespace-split tokens, not subwords.
    """
    return text.split()


# ── Vocabulary and frequency ──────────────────────────────────────────────────

def build_vocab(documents: list[str]) -> tuple[Counter, list[list[str]]]:
    """
    Build corpus-level vocabulary from a list of documents.

    Returns:
        vocab: Counter of {token: count} across entire corpus
        tokenized: list of per-document token lists (for TTR computation)
    """
    vocab: Counter = Counter()
    tokenized = []
    for doc in documents:
        tokens = whitespace_tokenize(doc)
        vocab.update(tokens)
        tokenized.append(tokens)
    return vocab, tokenized


def compute_ttr(tokenized: list[list[str]]) -> dict:
    """
    Type-token ratio at corpus level and per-document distribution.

    Corpus-level TTR: unique_types / total_tokens (affected by corpus size).
    Per-document TTR: distribution is more comparable across corpora of
    different sizes. Report both.
    """
    all_tokens: list[str] = []
    per_doc_ttrs = []
    for tokens in tokenized:
        all_tokens.extend(tokens)
        if tokens:
            per_doc_ttrs.append(len(set(tokens)) / len(tokens))

    corpus_types = len(set(all_tokens))
    corpus_tokens = len(all_tokens)
    corpus_ttr = corpus_types / corpus_tokens if corpus_tokens > 0 else 0.0

    return {
        "corpus_ttr": round(corpus_ttr, 6),
        "corpus_types": corpus_types,
        "corpus_tokens": corpus_tokens,
        "per_doc_ttr_mean": round(float(np.mean(per_doc_ttrs)), 6) if per_doc_ttrs else 0.0,
        "per_doc_ttr_std": round(float(np.std(per_doc_ttrs)), 6) if per_doc_ttrs else 0.0,
        "per_doc_ttr_median": round(float(np.median(per_doc_ttrs)), 6) if per_doc_ttrs else 0.0,
    }


# ── N-gram entropy ─────────────────────────────────────────────────────────────

def build_ngram_counts(
    tokenized: list[list[str]],
    n: int,
) -> Counter:
    """Build n-gram frequency counter from tokenized documents."""
    ngram_counter: Counter = Counter()
    for tokens in tokenized:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngram_counter[ngram] += 1
    return ngram_counter


def entropy_from_counts(counts: Counter) -> float:
    """
    Shannon entropy H = -sum(p * log2(p)) from a frequency Counter.
    Uses maximum-likelihood estimates (no smoothing — smoothing only in KL).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def compute_ngram_entropies(tokenized: list[list[str]]) -> dict:
    """Compute unigram, bigram, trigram Shannon entropy."""
    unigrams = build_ngram_counts(tokenized, 1)
    bigrams  = build_ngram_counts(tokenized, 2)
    trigrams = build_ngram_counts(tokenized, 3)

    return {
        "entropy_1gram": round(entropy_from_counts(unigrams), 6),
        "entropy_2gram": round(entropy_from_counts(bigrams), 6),
        "entropy_3gram": round(entropy_from_counts(trigrams), 6),
        "vocab_size_unigrams": len(unigrams),
        "vocab_size_bigrams":  len(bigrams),
        "vocab_size_trigrams": len(trigrams),
    }


# ── KL divergence baseline ────────────────────────────────────────────────────

def compute_kl_baseline_distributions(
    tokenized: list[list[str]],
) -> dict:
    """
    Serialize the baseline n-gram probability distributions for KL computation.

    In Phase 3, lexical.py will compare generated text's n-gram distribution
    against these baseline distributions using KL divergence.

    Distributions are stored as {ngram_str: probability} dicts with Laplace
    smoothing applied. The smoothed distribution is what gets stored — Phase 3
    must use the same smoothing when computing KL.

    Returns a dict with the top-N most frequent unigrams and trigrams
    (storing all n-grams would be too large; truncating to top-50k covers
    >99% of probability mass for any reasonable generated text sample).
    """
    TOP_N_UNIGRAMS = 50_000
    TOP_N_TRIGRAMS = 100_000
    LAPLACE_ALPHA  = 1  # add-1 smoothing

    unigrams = build_ngram_counts(tokenized, 1)
    trigrams = build_ngram_counts(tokenized, 3)

    # Smoothed unigram distribution (top-N only)
    vocab_size = len(unigrams)
    total_uni  = sum(unigrams.values())
    uni_dist = {}
    for ngram, count in unigrams.most_common(TOP_N_UNIGRAMS):
        smoothed = (count + LAPLACE_ALPHA) / (total_uni + LAPLACE_ALPHA * vocab_size)
        uni_dist[ngram[0]] = smoothed

    # Smoothed trigram distribution (top-N only)
    trigram_vocab = len(trigrams)
    total_tri     = sum(trigrams.values())
    tri_dist = {}
    for ngram, count in trigrams.most_common(TOP_N_TRIGRAMS):
        key = " ".join(ngram)
        smoothed = (count + LAPLACE_ALPHA) / (total_tri + LAPLACE_ALPHA * trigram_vocab)
        tri_dist[key] = smoothed

    return {
        "unigram_distribution": uni_dist,
        "trigram_distribution": tri_dist,
        "unigram_vocab_size":   vocab_size,
        "trigram_vocab_size":   trigram_vocab,
        "laplace_alpha":        LAPLACE_ALPHA,
        "unigram_total_tokens": total_uni,
        "trigram_total_tokens": total_tri,
    }


# ── Zipf fit ──────────────────────────────────────────────────────────────────

def fit_zipf(vocab: Counter) -> dict:
    """
    Fit Zipf's law to the corpus frequency distribution.

    Zipf's law: f(r) ∝ r^(-alpha), where r is rank and f is frequency.
    Fit is done on log-log scale via curve_fit.

    Expected alpha for natural English text: 0.8–1.2.
    Significant deviation from this range indicates corpus quality issues
    (too narrow domain → high alpha; preprocessing artifacts → low alpha).

    Returns:
        alpha: Zipf exponent (should be ~1.0 for natural text)
        r_squared: goodness of fit on log-log scale
        fit_diagnostic: pass/warn/fail based on alpha value
    """
    # Sort by frequency descending — rank 1 = most frequent
    sorted_counts = sorted(vocab.values(), reverse=True)
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=float)
    freqs = np.array(sorted_counts, dtype=float)

    # Fit on log scale (more numerically stable than raw scale)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    # Zipf model: log(f) = log(C) - alpha * log(r)
    def zipf_model(log_r, log_C, alpha):
        return log_C - alpha * log_r

    try:
        popt, _ = curve_fit(zipf_model, log_ranks, log_freqs)
        log_C, alpha = popt

        # R² on log-log scale
        log_freqs_pred = zipf_model(log_ranks, log_C, alpha)
        ss_res = np.sum((log_freqs - log_freqs_pred) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Diagnostic
        if 0.8 <= alpha <= 1.2:
            diagnostic = "pass"
        elif 0.6 <= alpha <= 1.5:
            diagnostic = "warn_outside_natural_range"
        else:
            diagnostic = "fail_not_natural_text"

        return {
            "zipf_alpha": round(float(alpha), 6),
            "zipf_log_C": round(float(log_C), 6),
            "zipf_r_squared": round(float(r_squared), 6),
            "fit_diagnostic": diagnostic,
            "n_unique_tokens": len(sorted_counts),
            "max_frequency": int(sorted_counts[0]) if sorted_counts else 0,
            "min_frequency": int(sorted_counts[-1]) if sorted_counts else 0,
            # Tail mass: fraction of vocabulary appearing ≤ 5 times
            "hapax_legomena_count": sum(1 for c in sorted_counts if c == 1),
            "rare_token_count_lte5": sum(1 for c in sorted_counts if c <= 5),
            "zipf_tail_mass_fraction": round(
                sum(1 for c in sorted_counts if c <= 5) / len(sorted_counts), 6
            ) if sorted_counts else 0.0,
        }

    except RuntimeError as e:
        return {"error": f"Zipf fit failed: {e}", "fit_diagnostic": "fail_curve_fit_error"}


# ── Main computation ──────────────────────────────────────────────────────────

def compute_lexical_baseline(
    documents: list[str],
    output_dir: Path,
    corpus_hash: str = "unknown",
) -> dict:
    """
    Compute all lexical baseline statistics and write to output_dir.

    Args:
        documents: list of cleaned document strings from the corpus
        output_dir: measurements/ directory to write JSON files into
        corpus_hash: SHA-256 of documents.jsonl for provenance tracking

    Returns:
        dict with all computed stats (same content as written to disk)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing lexical baseline on {len(documents):,} documents...")

    # Build vocab + tokenized representation
    vocab, tokenized = build_vocab(documents)
    print(f"  Vocabulary size: {len(vocab):,} types")

    # TTR
    ttr_stats = compute_ttr(tokenized)
    print(f"  Corpus TTR: {ttr_stats['corpus_ttr']:.4f}")

    # N-gram entropies
    entropy_stats = compute_ngram_entropies(tokenized)
    print(f"  Unigram entropy: {entropy_stats['entropy_1gram']:.4f} bits")
    print(f"  Trigram entropy: {entropy_stats['entropy_3gram']:.4f} bits")

    # Zipf fit
    zipf_stats = fit_zipf(vocab)
    if "error" not in zipf_stats:
        print(f"  Zipf alpha: {zipf_stats['zipf_alpha']:.4f} "
              f"(R²={zipf_stats['zipf_r_squared']:.4f}, {zipf_stats['fit_diagnostic']})")
    else:
        print(f"  [WARN] Zipf fit failed: {zipf_stats['error']}")

    # KL baseline distributions
    kl_dists = compute_kl_baseline_distributions(tokenized)
    print(f"  KL baseline: {len(kl_dists['unigram_distribution']):,} unigrams, "
          f"{len(kl_dists['trigram_distribution']):,} trigrams stored")

    # Assemble lexical_baseline.json
    lexical_baseline = {
        "_corpus_hash": corpus_hash,
        "_computed_at": __import__("datetime").datetime.now().isoformat(),
        **ttr_stats,
        **entropy_stats,
    }

    lexical_file = output_dir / "lexical_baseline.json"
    with open(lexical_file, "w") as f:
        json.dump(lexical_baseline, f, indent=2)
    print(f"  Written: {lexical_file}")

    # Write zipf_params.json separately (referenced independently in phase_criteria)
    zipf_file = output_dir / "zipf_params.json"
    with open(zipf_file, "w") as f:
        json.dump({"_corpus_hash": corpus_hash, **zipf_stats}, f, indent=2)
    print(f"  Written: {zipf_file}")

    # Write KL baseline distributions (large file — kept separate from scalar stats)
    kl_file = output_dir / "kl_baseline_distributions.json"
    with open(kl_file, "w") as f:
        json.dump({"_corpus_hash": corpus_hash, **kl_dists}, f)
    print(f"  Written: {kl_file} ({kl_file.stat().st_size / 1024:.1f} KB)")

    return {
        "lexical_baseline": lexical_baseline,
        "zipf_params": zipf_stats,
        "kl_distributions": kl_dists,
    }


# ── Self-test (no real corpus needed) ─────────────────────────────────────────

def run_self_test() -> bool:
    """
    Verify all lexical computations are directionally correct on synthetic data.

    Test A — Collapsed corpus (100 identical documents):
        TTR should be very low (near 0), entropy low, Zipf alpha high.

    Test B — Diverse corpus (100 varied documents):
        TTR should be higher, entropy higher, Zipf alpha closer to 1.0.
    """
    print("\n── Lexical Baseline Self-Test ────────────────────────────────────")
    all_passed = True

    # ── Test A: Collapsed corpus ──
    collapsed_docs = ["the cat sat on the mat"] * 100
    vocab_c, tokenized_c = build_vocab(collapsed_docs)
    ttr_c = compute_ttr(tokenized_c)
    ent_c = compute_ngram_entropies(tokenized_c)
    zipf_c = fit_zipf(vocab_c)

    passed = ttr_c["corpus_ttr"] < 0.05
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Collapsed TTR is low: "
          f"{ttr_c['corpus_ttr']:.4f} (expect < 0.05)")

    passed = ent_c["entropy_1gram"] < 3.0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Collapsed unigram entropy is low: "
          f"{ent_c['entropy_1gram']:.4f} bits (expect < 3.0)")

    # ── Test B: Diverse corpus ──
    # Generate lexically diverse documents using word grid
    import random
    rng = random.Random(42)
    wordlist = [f"word_{i}" for i in range(500)]
    diverse_docs = [
        " ".join(rng.choices(wordlist, k=50)) for _ in range(100)
    ]
    vocab_d, tokenized_d = build_vocab(diverse_docs)
    ttr_d = compute_ttr(tokenized_d)
    ent_d = compute_ngram_entropies(tokenized_d)

    passed = ttr_d["corpus_ttr"] > ttr_c["corpus_ttr"] * 10
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Diverse TTR >> Collapsed TTR: "
          f"{ttr_d['corpus_ttr']:.4f} vs {ttr_c['corpus_ttr']:.4f}")

    passed = ent_d["entropy_1gram"] > ent_c["entropy_1gram"]
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Diverse entropy > Collapsed entropy: "
          f"{ent_d['entropy_1gram']:.4f} > {ent_c['entropy_1gram']:.4f}")

    # ── Test C: Zipf fit on known distribution ──
    # Synthetic Zipf-distributed vocab: token_i has frequency ∝ 1/i
    vocab_zipf: Counter = Counter()
    for i in range(1, 501):
        vocab_zipf[f"w{i}"] = max(1, int(1000 / i))
    zipf_r = fit_zipf(vocab_zipf)
    passed = "error" not in zipf_r and 0.7 < zipf_r["zipf_alpha"] < 1.3
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Zipf fit on synthetic Zipf dist: "
          f"alpha={zipf_r.get('zipf_alpha', 'N/A'):.4f} (expect ~1.0)")

    # ── Test D: KL baseline distributions are non-empty ──
    kl = compute_kl_baseline_distributions(tokenized_d)
    passed = len(kl["unigram_distribution"]) > 0 and len(kl["trigram_distribution"]) > 0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] KL baseline non-empty: "
          f"{len(kl['unigram_distribution'])} unigrams, "
          f"{len(kl['trigram_distribution'])} trigrams")

    # ── Test E: Entropy is deterministic ──
    ent_d2 = compute_ngram_entropies(tokenized_d)
    passed = ent_d["entropy_1gram"] == ent_d2["entropy_1gram"]
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Entropy computation is deterministic")

    print()
    print("  ✓ LEXICAL SELF-TEST PASSED" if all_passed else "  ✗ LEXICAL SELF-TEST FAILED")
    return all_passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO lexical baseline computation")
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path(__file__).parent / "corpus",
                        help="Directory containing documents.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "measurements",
                        help="Output directory for measurement JSON files")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test on synthetic data (no corpus needed)")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    # Load corpus
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from phase1_baseline.corpus.ingest import load_corpus

    documents = load_corpus(args.corpus_dir)

    # Read corpus hash from manifest
    import json as _json
    manifest_file = args.corpus_dir / "manifest.json"
    corpus_hash = "unknown"
    if manifest_file.exists():
        with open(manifest_file) as f:
            corpus_hash = _json.load(f).get("corpus_sha256", "unknown")

    compute_lexical_baseline(documents, args.output_dir, corpus_hash)
    print("\nLexical baseline complete.")
