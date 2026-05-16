"""
MCO Phase 3 — Semantic Measurement Layer
==========================================
Measures semantic diversity using a FROZEN reference encoder.
Never fine-tune the encoder. It is the coordinate system.

Interface contract:
    measure(generated_samples, baseline_pack, **kwargs) -> dict[str, float]

Required kwargs:
    encoder: SentenceTransformer — must be frozen (eval + requires_grad=False)

No imports from sibling modules. Completely standalone.

Self-test:
    python phase3_measurements/layers/semantic.py
"""

from typing import Any
import numpy as np


def _assert_encoder_frozen(encoder) -> None:
    """Hard assertion: encoder must have no trainable parameters."""
    if hasattr(encoder, "parameters"):
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        assert len(trainable) == 0, (
            f"CRITICAL: encoder has {len(trainable)} trainable parameters. "
            "The reference encoder must be completely frozen. "
            "Load with model.eval() and requires_grad=False on all params."
        )


def _embed(texts: list[str], encoder, batch_size: int = 64) -> np.ndarray:
    """Encode texts with the frozen reference encoder."""
    _assert_encoder_frozen(encoder)
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 500,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


def _avg_pairwise_cosine_distance(emb: np.ndarray, seed: int = 42,
                                   max_samples: int = 500) -> float:
    """Average pairwise cosine distance over a random subsample."""
    n = len(emb)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_samples, n), replace=False)
    sub = emb[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = sub / norms
    sim_matrix = normed @ normed.T
    upper = np.triu_indices(len(normed), k=1)
    return float(np.mean(1.0 - sim_matrix[upper]))


def _pca_intrinsic_dim(emb: np.ndarray) -> float:
    """
    PCA proxy for intrinsic dimensionality.
    Returns number of components needed to explain 95% of variance.
    Note: systematically overestimates. TwoNN is preferred when skdim is available.
    """
    from sklearn.decomposition import PCA
    n_components = min(len(emb), emb.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(emb)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return float(np.searchsorted(cumvar, 0.95) + 1)


def _semantic_coverage(gen_emb_pca: np.ndarray,
                        baseline_pca: np.ndarray,
                        n_components: int = 10) -> float:
    """
    Fraction of baseline semantic modes covered by generated samples.
    Fits a GMM on baseline embeddings, checks how many modes have
    at least one generated sample within 1.5σ.
    """
    from sklearn.mixture import GaussianMixture

    n_comp = min(n_components, len(baseline_pca) // 10, len(gen_emb_pca))
    if n_comp < 2:
        return 0.0

    gmm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                          random_state=42, max_iter=200)
    gmm.fit(baseline_pca)

    n_covered = 0
    for i in range(n_comp):
        center = gmm.means_[i]
        std = float(np.sqrt(np.mean(gmm.covariances_[i])))
        dists = np.linalg.norm(gen_emb_pca - center, axis=1)
        if np.any(dists < 1.5 * std):
            n_covered += 1
    return n_covered / n_comp


def measure(
    generated_samples: list[str],
    baseline_pack: dict[str, Any],
    encoder=None,
    seed: int = 42,
    **kwargs,
) -> dict[str, float]:
    """
    Compute semantic diversity measurements.

    Args:
        generated_samples: list of generated text strings
        baseline_pack: dict containing:
            - pca: fitted sklearn PCA transform (from Phase 1)
            - embeddings_pca: (N, 20) array of baseline PCA embeddings
            - semantic: dict with avg_pairwise_cosine_distance, intrinsic_dimensionality
        encoder: FROZEN SentenceTransformer (required)

    Returns dict with keys:
        avg_pairwise_cosine_dist, intrinsic_dimensionality, semantic_coverage,
        baseline_avg_cosine_dist, cosine_dist_rel_change
    """
    if encoder is None:
        raise ValueError(
            "encoder is required for semantic measurement. "
            "Pass a frozen SentenceTransformer via measure(..., encoder=enc)"
        )
    _assert_encoder_frozen(encoder)

    pca        = baseline_pack["pca"]
    baseline_pca = baseline_pack["embeddings_pca"]
    sem_baseline = baseline_pack.get("semantic", {})

    # Embed generated samples
    gen_emb     = _embed(generated_samples, encoder)
    gen_emb_pca = pca.transform(gen_emb)

    cos_dist   = _avg_pairwise_cosine_distance(gen_emb, seed=seed)
    id_val     = _pca_intrinsic_dim(gen_emb)
    coverage   = _semantic_coverage(gen_emb_pca, baseline_pca)

    baseline_cos = sem_baseline.get("avg_pairwise_cosine_distance",
                   sem_baseline.get("avg_cosine_distance", None))
    rel_change = ((cos_dist - baseline_cos) / baseline_cos) \
        if baseline_cos and baseline_cos > 0 else None

    return {
        "avg_pairwise_cosine_dist":   round(float(cos_dist), 6),
        "intrinsic_dimensionality":   round(float(id_val), 4),
        "semantic_coverage":          round(float(coverage), 6),
        "baseline_avg_cosine_dist":   baseline_cos,
        "cosine_dist_rel_change":     round(rel_change, 6) if rel_change is not None else None,
    }


# ── Self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Semantic Layer Self-Test ─────────────────────────────")
    passed = failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        marker = "[PASS]" if condition else "[FAIL]"
        print(f"  {marker} {name}{' — ' + detail if detail else ''}")
        if condition:
            passed += 1
        else:
            failed += 1

    # Build a synthetic encoder that produces fixed embeddings
    class _FakeEncoder:
        """Frozen fake encoder for self-test — no real model needed."""
        def encode(self, texts, **kwargs):
            rng = np.random.default_rng(len(texts[0]) if texts else 0)
            return rng.random((len(texts), 32)).astype(np.float32)
        def parameters(self):
            return iter([])  # no parameters → frozen assertion passes

    encoder = _FakeEncoder()

    # Collapsed: identical texts → nearly identical embeddings
    class _CollapsedEncoder:
        def encode(self, texts, **kwargs):
            base = np.ones((1, 32), dtype=np.float32) * 0.5
            noise = np.random.default_rng(42).random((len(texts), 32)) * 1e-6
            return (base + noise).astype(np.float32)
        def parameters(self):
            return iter([])

    class _DiverseEncoder:
        def encode(self, texts, **kwargs):
            rng = np.random.default_rng(12345)
            return rng.random((len(texts), 32)).astype(np.float32)
        def parameters(self):
            return iter([])

    from sklearn.decomposition import PCA as _PCA

    # Make baseline pack with PCA fitted on diverse data
    baseline_data = np.random.default_rng(0).random((200, 32)).astype(np.float32)
    pca_fit = _PCA(n_components=10, random_state=42)
    pca_fit.fit(baseline_data)
    baseline_pca = pca_fit.transform(baseline_data)
    pack = {
        "pca": pca_fit,
        "embeddings_pca": baseline_pca,
        "semantic": {"avg_pairwise_cosine_distance": 0.5},
    }

    texts_c = ["the cat sat"] * 100
    texts_d = [f"document number {i} with unique content" for i in range(100)]

    r_c = measure(texts_c, pack, encoder=_CollapsedEncoder())
    r_d = measure(texts_d, pack, encoder=_DiverseEncoder())

    check("Identical embeddings → dist≈0",
          r_c["avg_pairwise_cosine_dist"] < 0.01,
          f"{r_c['avg_pairwise_cosine_dist']:.6f}")
    check("Diverse > Identical cosine dist",
          r_d["avg_pairwise_cosine_dist"] > r_c["avg_pairwise_cosine_dist"],
          f"{r_d['avg_pairwise_cosine_dist']:.4f} > {r_c['avg_pairwise_cosine_dist']:.4f}")
    check("Diverse coverage >= Identical coverage",
          r_d["semantic_coverage"] >= r_c["semantic_coverage"],
          f"{r_d['semantic_coverage']:.2f} >= {r_c['semantic_coverage']:.2f}")
    check("Intrinsic dim > 5 for diverse",
          r_d["intrinsic_dimensionality"] > 5,
          f"dim={r_d['intrinsic_dimensionality']}")

    print(f"\n  {'✓ SEMANTIC SELF-TEST PASSED' if failed == 0 else '✗ SEMANTIC SELF-TEST FAILED'} "
          f"({passed}/{passed+failed})")
