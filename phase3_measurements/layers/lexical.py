"""
MCO Phase 3 — Lexical Measurement Layer
=========================================
Measures n-gram entropy, type-token ratio, Zipf tail, and KL divergence
from the human baseline distribution.

Interface contract:
    measure(generated_samples, baseline_pack, **kwargs) -> dict[str, float]

No imports from sibling modules. Completely standalone.

Self-test:
    python phase3_measurements/layers/lexical.py
"""

import math
from collections import Counter
from typing import Any


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def _ngram_entropy(tokenized: list[list[str]], n: int) -> float:
    counts: Counter = Counter()
    for toks in tokenized:
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i: i + n])] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _fit_zipf_alpha(sorted_counts: list[int]) -> float:
    """Fit Zipf exponent via log-log OLS. Returns alpha (positive = heavy tail)."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    if len(sorted_counts) < 10:
        return 0.0
    import numpy as np
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=float)
    freqs = np.array(sorted_counts, dtype=float)
    A = np.column_stack([np.ones_like(ranks), np.log(ranks)])
    result = np.linalg.lstsq(A, np.log(freqs + 1e-9), rcond=None)
    return float(-result[0][1])


def compute_kl_divergence(
    generated_texts: list[str],
    baseline_unigram_dist: dict[str, float],
    laplace_alpha: float = 1.0,
) -> float:
    """
    KL(generated || baseline): how much generated text diverges from human baseline.

    Direction is critical for correct collapse detection:
    - KL(generated || baseline): HIGH when model generates concentrated text
      (e.g., all "the" → gen_p("the")≈1, baseline_p("the")≈0.05 → KL≈4.3)
    - KL(baseline || generated): LOW for collapsed text (wrong direction for
      collapse detection — collapsed text happens to contain baseline words)

    As collapse proceeds: KL(generated||baseline) INCREASES — model generates
    text that deviates more and more from what humans write.
    """
    gen_counts: Counter = Counter()
    for t in generated_texts:
        gen_counts.update(whitespace_tokenize(t))
    total = sum(gen_counts.values())
    if total == 0:
        return float("inf")

    baseline_total = sum(baseline_unigram_dist.values())
    vocab_size = len(baseline_unigram_dist)

    def bp(word: str) -> float:
        raw = baseline_unigram_dist.get(word, 0.0)
        return (raw * baseline_total + laplace_alpha) / (
            baseline_total + laplace_alpha * (vocab_size + 1)
        )

    kl = 0.0
    for word, count in gen_counts.items():
        gen_p = count / total
        b = bp(word)
        if gen_p > 0 and b > 0:
            kl += gen_p * math.log2(gen_p / b)
    return max(0.0, kl)


def measure(
    generated_samples: list[str],
    baseline_pack: dict[str, Any],
    **kwargs,
) -> dict[str, float]:
    """
    Compute lexical measurements on generated_samples against the baseline.

    Args:
        generated_samples: list of generated text strings
        baseline_pack: dict containing:
            - lexical: dict with corpus_ttr, entropy_1gram, entropy_3gram, zipf_alpha
            - kl_distributions: dict with unigram_distribution, laplace_alpha

    Returns dict with keys:
        ttr, entropy_1gram, entropy_2gram, entropy_3gram,
        kl_div_1gram, zipf_alpha, zipf_tail_mass,
        baseline_entropy_1gram, entropy_1gram_rel_change
    """
    if not generated_samples:
        return {k: 0.0 for k in [
            "ttr", "entropy_1gram", "entropy_2gram", "entropy_3gram",
            "kl_div_1gram", "zipf_alpha", "zipf_tail_mass",
        ]}

    tokenized = [whitespace_tokenize(s) for s in generated_samples]
    all_tokens: list[str] = [t for toks in tokenized for t in toks]

    if not all_tokens:
        return {k: 0.0 for k in [
            "ttr", "entropy_1gram", "entropy_2gram", "entropy_3gram",
            "kl_div_1gram", "zipf_alpha", "zipf_tail_mass",
        ]}

    vocab = Counter(all_tokens)
    total_tokens = len(all_tokens)
    ttr = len(vocab) / total_tokens

    entropy_1g = _ngram_entropy(tokenized, 1)
    entropy_2g = _ngram_entropy(tokenized, 2)
    entropy_3g = _ngram_entropy(tokenized, 3)

    # KL divergence from baseline (correct direction: generated || baseline)
    kl_dists = baseline_pack.get("kl_distributions", {})
    uni_baseline = kl_dists.get("unigram_distribution", {})
    laplace = float(kl_dists.get("laplace_alpha", 1.0))
    kl_div_1g = compute_kl_divergence(generated_samples, uni_baseline, laplace) \
        if uni_baseline else 0.0

    # Zipf fit
    sorted_counts = sorted(vocab.values(), reverse=True)
    zipf_alpha = _fit_zipf_alpha(sorted_counts)
    hapax_count = sum(1 for c in sorted_counts if c == 1)
    zipf_tail_mass = hapax_count / len(sorted_counts) if sorted_counts else 0.0

    # Relative change from baseline
    lex_baseline = baseline_pack.get("lexical", {})
    baseline_ent = lex_baseline.get("entropy_1gram", None)
    rel_change = ((entropy_1g - baseline_ent) / baseline_ent) \
        if baseline_ent and baseline_ent > 0 else None

    return {
        "ttr":                      round(ttr, 6),
        "entropy_1gram":            round(entropy_1g, 6),
        "entropy_2gram":            round(entropy_2g, 6),
        "entropy_3gram":            round(entropy_3g, 6),
        "kl_div_1gram":             round(kl_div_1g, 6),
        "zipf_alpha":               round(zipf_alpha, 6),
        "zipf_tail_mass":           round(zipf_tail_mass, 6),
        "baseline_entropy_1gram":   baseline_ent,
        "entropy_1gram_rel_change": round(rel_change, 6) if rel_change is not None else None,
    }


# ── Self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("── Lexical Layer Self-Test ──────────────────────────────")
    passed = failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        marker = "[PASS]" if condition else "[FAIL]"
        print(f"  {marker} {name}{' — ' + detail if detail else ''}")
        if condition:
            passed += 1
        else:
            failed += 1

    empty_pack = {"lexical": {"entropy_1gram": 8.0}, "kl_distributions": {}}

    collapsed = ["the the the the the the the"] * 200
    rng = random.Random(42)
    words = [f"word_{i}" for i in range(1000)]
    diverse = [" ".join(rng.choices(words, k=30)) for _ in range(200)]

    r_c = measure(collapsed, empty_pack)
    r_d = measure(diverse, empty_pack)

    check("Collapsed TTR < 0.05", r_c["ttr"] < 0.05, f"TTR={r_c['ttr']:.4f}")
    check("Collapsed entropy < 1.0", r_c["entropy_1gram"] < 1.0,
          f"H1={r_c['entropy_1gram']:.4f}")
    check("Diverse TTR > Collapsed TTR",
          r_d["ttr"] > r_c["ttr"] * 5,
          f"{r_d['ttr']:.4f} > {r_c['ttr']:.4f}")
    check("Diverse entropy > Collapsed entropy",
          r_d["entropy_1gram"] > r_c["entropy_1gram"],
          f"{r_d['entropy_1gram']:.4f} > {r_c['entropy_1gram']:.4f}")

    # KL test with a real baseline distribution
    baseline_with_dist = {
        "lexical": {"entropy_1gram": 8.0},
        "kl_distributions": {
            "unigram_distribution": {
                "the": 0.05, "cat": 0.02, "sat": 0.02, "on": 0.03,
                "mat": 0.01, "a": 0.04, "in": 0.03,
            },
            "laplace_alpha": 1,
        },
    }
    r_c2 = measure(collapsed, baseline_with_dist)
    r_d2 = measure(diverse, baseline_with_dist)
    check("Collapsed KL > Diverse KL (correct direction)",
          r_c2["kl_div_1gram"] > r_d2["kl_div_1gram"],
          f"collapsed={r_c2['kl_div_1gram']:.4f} > diverse={r_d2['kl_div_1gram']:.4f}")

    # Determinism
    r_d3 = measure(diverse, empty_pack)
    check("Deterministic", r_d["entropy_1gram"] == r_d3["entropy_1gram"])

    print(f"\n  {'✓ LEXICAL SELF-TEST PASSED' if failed == 0 else '✗ LEXICAL SELF-TEST FAILED'} "
          f"({passed}/{passed+failed})")
