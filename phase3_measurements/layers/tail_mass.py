"""
MCO Phase 3 — Tail Mass Measurement Layer
==========================================
Measures the fraction of generated samples that fall in the tail of the
human baseline density. The tail threshold is FROZEN from Phase 1 — it
never changes between generations.

Interface contract:
    measure(generated_samples, baseline_pack, **kwargs) -> dict[str, float]

Required kwargs:
    encoder: FROZEN SentenceTransformer

No imports from sibling modules. Completely standalone.

Self-test:
    python phase3_measurements/layers/tail_mass.py
"""

from typing import Any
import numpy as np


def _assert_encoder_frozen(encoder) -> None:
    if hasattr(encoder, "parameters"):
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        assert len(trainable) == 0, (
            "CRITICAL: encoder has trainable parameters. "
            "Tail mass requires a frozen reference encoder."
        )


def measure(
    generated_samples: list[str],
    baseline_pack: dict[str, Any],
    encoder=None,
    **kwargs,
) -> dict[str, float]:
    """
    Compute tail mass measurements.

    The tail threshold was set at the 5th percentile of baseline log-likelihood
    in Phase 1 and is PERMANENTLY FROZEN. Never refit or adjust it.

    As collapse proceeds:
    - mean_log_likelihood INCREASES (model generates more "central" samples)
    - std_log_likelihood DECREASES (less variance — fewer tail samples)
    - tail_mass_fraction DECREASES (the collapse signal)

    Args:
        generated_samples: list of generated text strings
        baseline_pack: dict containing:
            - kde: fitted sklearn KernelDensity (from Phase 1)
            - tail_threshold: float (5th percentile, FROZEN)
            - pca: fitted PCA transform
            - embeddings_pca: baseline PCA embeddings for reference comparison
        encoder: FROZEN SentenceTransformer

    Returns dict with keys:
        mean_log_likelihood, std_log_likelihood, tail_mass_fraction,
        tail_threshold, baseline_tail_fraction, tail_fraction_rel_change
    """
    if encoder is None:
        raise ValueError(
            "encoder is required for tail mass measurement. "
            "Pass a frozen SentenceTransformer via measure(..., encoder=enc)"
        )
    _assert_encoder_frozen(encoder)

    kde            = baseline_pack["kde"]
    threshold      = float(baseline_pack["tail_threshold"])
    pca            = baseline_pack["pca"]
    baseline_pca   = baseline_pack["embeddings_pca"]

    # Embed and project generated samples
    gen_emb = encoder.encode(
        generated_samples,
        batch_size=64,
        show_progress_bar=len(generated_samples) > 500,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    gen_pca = pca.transform(gen_emb)

    # Score under baseline density
    gen_ll = kde.score_samples(gen_pca)
    mean_ll   = float(np.mean(gen_ll))
    std_ll    = float(np.std(gen_ll))
    tail_frac = float(np.mean(gen_ll < threshold))

    # Baseline reference (same threshold applied to baseline embeddings)
    baseline_ll   = kde.score_samples(baseline_pca)
    baseline_tail = float(np.mean(baseline_ll < threshold))

    # Relative change (negative = fewer tail samples = collapse signal)
    rel_change = ((tail_frac - baseline_tail) / baseline_tail) \
        if baseline_tail > 0 else None

    return {
        "mean_log_likelihood":       round(mean_ll, 6),
        "std_log_likelihood":        round(std_ll, 6),
        "tail_mass_fraction":        round(tail_frac, 6),
        "tail_threshold":            round(threshold, 6),
        "baseline_tail_fraction":    round(baseline_tail, 6),
        "tail_fraction_rel_change":  round(rel_change, 6) if rel_change is not None else None,
    }


# ── Self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Tail Mass Layer Self-Test ────────────────────────────")
    passed = failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        marker = "[PASS]" if condition else "[FAIL]"
        print(f"  {marker} {name}{' — ' + detail if detail else ''}")
        if condition:
            passed += 1
        else:
            failed += 1

    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA as _PCA

    rng = np.random.default_rng(42)

    # Baseline: diverse 32-dim embeddings
    baseline_raw = rng.random((500, 32)).astype(np.float32)
    pca_fit = _PCA(n_components=10, random_state=42)
    pca_fit.fit(baseline_raw)
    baseline_pca = pca_fit.transform(baseline_raw)

    # Fit KDE on 60% of baseline
    n_train = int(0.6 * len(baseline_pca))
    kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde.fit(baseline_pca[:n_train])

    # Tail threshold: 5th percentile on held-out 40%
    eval_ll = kde.score_samples(baseline_pca[n_train:])
    threshold = float(np.percentile(eval_ll, 5))

    pack = {
        "kde": kde,
        "tail_threshold": threshold,
        "pca": pca_fit,
        "embeddings_pca": baseline_pca,
    }

    # Collapsed encoder: all samples at distribution center
    class _CollapsedEncoder:
        def encode(self, texts, **kwargs):
            center = np.mean(baseline_raw, axis=0, keepdims=True)
            noise  = np.random.default_rng(0).random((len(texts), 32)) * 0.01
            return (center + noise).astype(np.float32)
        def parameters(self): return iter([])

    # Baseline encoder: samples drawn from same distribution as baseline
    class _BaselineEncoder:
        def encode(self, texts, **kwargs):
            return np.random.default_rng(99).random((len(texts), 32)).astype(np.float32)
        def parameters(self): return iter([])

    texts = ["text"] * 200

    r_b = measure(texts, pack, encoder=_BaselineEncoder())
    r_c = measure(texts, pack, encoder=_CollapsedEncoder())

    check("Baseline tail fraction ~5%",
          0.02 <= r_b["baseline_tail_fraction"] <= 0.10,
          f"{r_b['baseline_tail_fraction']:.3f}")
    check("Collapsed tail < Baseline tail",
          r_c["tail_mass_fraction"] < r_b["baseline_tail_fraction"],
          f"{r_c['tail_mass_fraction']:.3f} < {r_b['baseline_tail_fraction']:.3f}")
    check("Collapsed mean LL > Baseline mean LL",
          r_c["mean_log_likelihood"] > r_b["mean_log_likelihood"],
          f"{r_c['mean_log_likelihood']:.3f} > {r_b['mean_log_likelihood']:.3f}")

    print(f"\n  {'✓ TAIL MASS SELF-TEST PASSED' if failed == 0 else '✗ TAIL MASS SELF-TEST FAILED'} "
          f"({passed}/{passed+failed})")
