"""
MCO Phase 1 — Semantic Baseline
=================================
Encodes the human corpus with the frozen reference encoder, fits PCA,
computes intrinsic dimensionality and semantic coverage statistics.

Produces:
    measurements/semantic_baseline.json  — intrinsic dim, cosine dist, coverage
    measurements/pca_transform.pkl       — fitted PCA (fit ONCE, never refit)
    measurements/semantic_embeddings.npy — baseline embedding matrix (sampled)

Critical constraints:
    - Encoder is loaded with eval() + requires_grad=False. No exceptions.
    - PCA is fit HERE and serialized. Phase 3 loads and applies it — never refits.
    - PCA sign convention must be asserted at load time (see assert_pca_sign()).
    - All downstream measurements use the SAME PCA projection.

Self-test (no corpus or GPU needed):
    python phase1_baseline/embeddings.py --self-test
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


# ── Encoder loading ───────────────────────────────────────────────────────────

def load_frozen_encoder(model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the reference sentence encoder with all gradients disabled.

    This encoder is the measurement coordinate system. It must never be
    fine-tuned, updated, or replaced after Phase 1. Any code that passes
    this model to an optimizer is a critical bug.

    Returns:
        model: SentenceTransformer with eval() + requires_grad=False
        embedding_dim: int (384 for all-MiniLM-L6-v2)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers==2.6.1"
        )

    model = SentenceTransformer(model_id)

    # Freeze — belt and suspenders
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Assert no gradients are attached — runtime guard
    _assert_encoder_frozen(model)

    embedding_dim = model.get_sentence_embedding_dimension()
    return model, embedding_dim


def _assert_encoder_frozen(model) -> None:
    """
    Runtime assertion: verify no parameters require gradients.
    Call this at the start of every measurement session that uses the encoder.
    If this raises, something upstream is training the encoder — critical bug.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            raise RuntimeError(
                f"CRITICAL: Encoder parameter '{name}' has requires_grad=True. "
                f"The reference encoder must never be trained. "
                f"This invalidates all cross-generation comparisons."
            )


def encode_corpus(
    model,
    documents: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode documents with the frozen reference encoder.

    Returns:
        embeddings: np.ndarray of shape (n_documents, embedding_dim), float32
    """
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize manually where needed
    )
    return embeddings.astype(np.float32)


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_pca(embeddings: np.ndarray, n_components: int = 20) -> PCA:
    """
    Fit PCA on baseline embeddings. Returns the fitted PCA object.

    This PCA is fit ONCE on Phase 1 baseline data and serialized.
    Phase 3 measurement modules load and apply it — they never refit.
    Refitting PCA on generated data would change the coordinate system
    and make cross-generation comparisons meaningless.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    return pca


def assert_pca_sign(pca: PCA) -> PCA:
    """
    Enforce PCA sign convention: first element of first component must be positive.

    PCA eigenvectors are arbitrary up to sign. If you load a saved PCA and the
    sign has flipped (possible across numpy/sklearn versions), all projections
    will be in the wrong half-space. This assertion detects and corrects it.

    Call this every time you load pca_transform.pkl before using it.
    """
    if pca.components_[0][0] < 0:
        pca.components_[0] = -pca.components_[0]
    return pca


def save_pca(pca: PCA, path: Path, corpus_hash: str = "unknown") -> None:
    """
    Serialize the fitted PCA with metadata.
    Saves as pickle (sklearn object) alongside a JSON sidecar with key params.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Assert sign convention before saving
    pca = assert_pca_sign(pca)

    with open(path, "wb") as f:
        pickle.dump(pca, f, protocol=4)  # protocol=4 for Python 3.8+ compat

    # JSON sidecar — allows verifying PCA params without unpickling
    sidecar = {
        "_corpus_hash": corpus_hash,
        "_sklearn_version": _get_sklearn_version(),
        "n_components": int(pca.n_components_),
        "n_features_in": int(pca.n_features_in_),
        "explained_variance_ratio_sum": round(
            float(pca.explained_variance_ratio_.sum()), 6
        ),
        "first_component_first_element": float(pca.components_[0][0]),
        "sign_convention": "first_component[0] > 0 (enforced at save and load)",
    }
    sidecar_path = path.with_suffix(".json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def load_pca(path: Path) -> PCA:
    """
    Load the saved PCA transform, assert sign convention, return ready-to-use PCA.
    Always use this function — never unpickle directly.
    """
    with open(path, "rb") as f:
        pca = pickle.load(f)

    pca = assert_pca_sign(pca)

    # Verify sidecar if present
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        current_sklearn = _get_sklearn_version()
        saved_sklearn = sidecar.get("_sklearn_version", "unknown")
        if current_sklearn != saved_sklearn:
            print(f"[WARN] sklearn version mismatch: PCA saved with {saved_sklearn}, "
                  f"loading with {current_sklearn}. Results may differ slightly.")

    return pca


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "unknown"


# ── Semantic statistics ────────────────────────────────────────────────────────

def compute_avg_pairwise_cosine_distance(
    embeddings: np.ndarray,
    sample_size: int = 500,
    seed: int = 42,
) -> float:
    """
    Estimate average pairwise cosine distance on a random subset.

    Full pairwise computation is O(n²) — prohibitive for 5k docs.
    Sample 500 docs, compute pairwise distances, report mean.
    The estimate is stable at sample_size=500.

    Returns:
        mean cosine distance (higher = more diverse, range [0, 2])
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    if n > sample_size:
        indices = rng.choice(n, size=sample_size, replace=False)
        subset = embeddings[indices]
    else:
        subset = embeddings

    # Normalize to unit vectors
    norms = np.linalg.norm(subset, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    subset_normalized = subset / norms

    # Cosine similarity matrix
    cos_sim = subset_normalized @ subset_normalized.T

    # Cosine distance = 1 - cosine similarity
    # Take upper triangle (exclude diagonal)
    n_sub = len(subset_normalized)
    upper_tri_indices = np.triu_indices(n_sub, k=1)
    cos_distances = 1 - cos_sim[upper_tri_indices]

    return float(np.mean(cos_distances))


def compute_intrinsic_dimensionality(
    embeddings: np.ndarray,
    sample_size: int = 1000,
    seed: int = 42,
) -> float:
    """
    Estimate intrinsic dimensionality using the Two-Nearest-Neighbors (TwoNN) estimator.

    TwoNN measures the ratio of distances to the first and second nearest neighbors.
    For a d-dimensional manifold, this ratio follows a Pareto distribution with
    parameter d. The estimator is parameter-free and robust.

    Expected range for a diverse natural language corpus: > 10.
    Low intrinsic dim (<8) means the corpus is semantically compressed
    (topic-skewed), which will limit the sensitivity of collapse detection.

    Reference: Facco et al. (2017), doi:10.1038/s41598-017-11873-y
    """
    try:
        import skdim
        rng = np.random.default_rng(seed)
        n = len(embeddings)
        if n > sample_size:
            indices = rng.choice(n, size=sample_size, replace=False)
            subset = embeddings[indices]
        else:
            subset = embeddings

        estimator = skdim.id.TwoNN()
        id_estimate = estimator.fit(subset).dimension_
        return float(id_estimate)

    except ImportError:
        print("[WARN] skdim not installed — using PCA-based dimensionality proxy. "
              "Install skdim for accurate TwoNN estimate: pip install skdim==0.3.4")
        return _pca_dimensionality_proxy(embeddings)


def _pca_dimensionality_proxy(embeddings: np.ndarray) -> float:
    """
    Fallback intrinsic dimensionality estimate: number of PCA components
    needed to explain 95% of variance. Less accurate than TwoNN but
    requires no additional dependencies.
    """
    n_max = min(len(embeddings), embeddings.shape[1])
    pca_full = PCA(n_components=n_max, random_state=42)
    pca_full.fit(embeddings)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    return float(np.searchsorted(cumvar, 0.95) + 1)


def compute_semantic_coverage(*args, **kwargs):
    """
    Semantic coverage computation has been moved to Phase 3.

    This function does NOT belong in Phase 1. It takes `generated_embeddings`
    as a parameter, which means it is a Phase 3 measurement function — it
    compares generated text against the baseline. Keeping it here would create
    a cross-phase import dependency that violates module independence.

    Use: phase3_measurements/layers/semantic.py → compute_semantic_coverage()
    The GMM is fit there using the baseline embeddings loaded from the
    reference pack, and applied to each generation's embeddings.

    Raises RuntimeError to catch any accidental calls during Phase 1.
    """
    raise RuntimeError(
        "compute_semantic_coverage() belongs in Phase 3, not Phase 1. "
        "Import from phase3_measurements/layers/semantic.py instead."
    )


# ── Main computation ───────────────────────────────────────────────────────────

def compute_semantic_baseline(
    documents: list[str],
    output_dir: Path,
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_pca_components: int = 20,
    n_embedding_sample: int = 2000,
    corpus_hash: str = "unknown",
    seed: int = 42,
) -> dict:
    """
    Encode corpus, fit PCA, compute semantic statistics, serialize all artifacts.

    The embedding matrix saved to disk is a random sample (n_embedding_sample)
    of the full corpus — storing all 5k embeddings is fine but the sample is
    sufficient for KDE fitting and is faster to load.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Loading frozen encoder: {model_id}")
    model, embedding_dim = load_frozen_encoder(model_id)
    print(f"  Embedding dim: {embedding_dim}")

    print(f"Encoding {len(documents):,} documents (batch_size=32)...")
    embeddings = encode_corpus(model, documents, batch_size=32)
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"Fitting PCA ({n_pca_components} components)...")
    pca = fit_pca(embeddings, n_components=n_pca_components)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.3f}")

    # Project to PCA space
    embeddings_pca = pca.transform(embeddings)

    print("Computing intrinsic dimensionality (TwoNN)...")
    intrinsic_dim = compute_intrinsic_dimensionality(embeddings, seed=seed)
    print(f"  Intrinsic dimensionality: {intrinsic_dim:.2f}")

    print("Computing average pairwise cosine distance...")
    avg_cos_dist = compute_avg_pairwise_cosine_distance(embeddings, seed=seed)
    print(f"  Avg pairwise cosine distance: {avg_cos_dist:.4f}")

    # Sample raw (384-dim) embeddings for storage — these are large.
    # 5k × 384 × 4 bytes = 7.7MB. Storing a 2k sample saves space without
    # losing information needed downstream (raw embeddings are not used for
    # KDE or tail mass — only PCA-projected embeddings are).
    #
    # PCA-projected embeddings are NEVER subsampled here.
    # 5k × 20 × 4 bytes = 400KB — trivially cheap to store all of them.
    # Subsampling before KDE degrades the tail threshold estimate because:
    #   - 60/40 split on 2k gives 800 held-out points for a 20-dim threshold
    #   - 5th percentile of 800 points has meaningful variance
    #   - Threshold instability biases Phase 3 collapse detection
    # Always save all PCA embeddings.
    n = len(embeddings)
    if n > n_embedding_sample:
        raw_sample_idx = rng.choice(n, size=n_embedding_sample, replace=False)
        raw_sample_idx.sort()
        embeddings_sample = embeddings[raw_sample_idx]
    else:
        embeddings_sample = embeddings

    # PCA embeddings: all documents, no subsampling
    embeddings_pca_all = embeddings_pca  # shape: (n_documents, n_pca_components)

    # Save artifacts
    pca_path = output_dir / "pca_transform.pkl"
    save_pca(pca, pca_path, corpus_hash)
    print(f"  Written: {pca_path}")

    emb_path = output_dir / "semantic_embeddings.npy"
    np.save(emb_path, embeddings_sample, allow_pickle=False)
    print(f"  Written: {emb_path} (shape={embeddings_sample.shape}, raw 384-dim sample)")

    emb_pca_path = output_dir / "semantic_embeddings_pca.npy"
    np.save(emb_pca_path, embeddings_pca_all, allow_pickle=False)
    print(f"  Written: {emb_pca_path} (shape={embeddings_pca_all.shape}, ALL docs, no subsampling)")

    semantic_baseline = {
        "_corpus_hash": corpus_hash,
        "_computed_at": __import__("datetime").datetime.now().isoformat(),
        "encoder_id": model_id,
        "embedding_dim": embedding_dim,
        "n_documents_encoded": len(documents),
        "n_raw_embeddings_stored": len(embeddings_sample),
        "n_pca_embeddings_stored": len(embeddings_pca_all),
        "pca_subsampled": False,
        "raw_subsampled": len(embeddings_sample) < len(embeddings),
        "pca_components": n_pca_components,
        "pca_explained_variance_ratio": round(float(explained), 6),
        "intrinsic_dimensionality": round(intrinsic_dim, 4),
        "avg_pairwise_cosine_distance": round(avg_cos_dist, 6),
        # ── Geometry note ──────────────────────────────────────────────────
        # PCA and KDE operate in Euclidean space on unnormalized embeddings.
        # avg_pairwise_cosine_distance normalizes vectors before computing
        # angular distance. These are intentionally different signals:
        # Euclidean/PCA captures variance structure and density in the
        # original embedding space; cosine captures semantic dispersion
        # independent of embedding magnitude.
        # ──────────────────────────────────────────────────────────────────
        "geometry_note": "PCA/KDE: Euclidean on unnormalized embeds. Cosine: angular on normalized. Intentionally different signals.",
        "sign_convention": "pca.components_[0][0] > 0",
    }

    semantic_file = output_dir / "semantic_baseline.json"
    with open(semantic_file, "w") as f:
        json.dump(semantic_baseline, f, indent=2)
    print(f"  Written: {semantic_file}")

    return {
        "semantic_baseline": semantic_baseline,
        "embeddings": embeddings,
        "embeddings_pca": embeddings_pca,
        "pca": pca,
    }


# ── Self-test (synthetic data, no encoder needed) ─────────────────────────────

def run_self_test() -> bool:
    """
    Test all geometric computations on synthetic embeddings.
    Does NOT require sentence-transformers to be installed.
    """
    print("\n── Semantic Baseline Self-Test ───────────────────────────────────")
    all_passed = True
    rng = np.random.default_rng(42)

    # ── Test A: PCA fits and sign convention holds ──
    fake_embeddings = rng.standard_normal((200, 384)).astype(np.float32)
    pca = fit_pca(fake_embeddings, n_components=20)
    pca = assert_pca_sign(pca)

    passed = pca.components_[0][0] > 0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] PCA sign convention: "
          f"components_[0][0] = {pca.components_[0][0]:.4f} > 0")

    passed = pca.n_components_ == 20
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] PCA n_components = {pca.n_components_}")

    # ── Test B: Cosine distance — identical → 0, diverse → high ──
    identical = np.ones((50, 384), dtype=np.float32)  # all same
    dist_identical = compute_avg_pairwise_cosine_distance(identical, seed=42)
    passed = dist_identical < 0.01
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Identical embeddings → dist≈0: "
          f"{dist_identical:.6f}")

    diverse = rng.standard_normal((50, 384)).astype(np.float32)
    dist_diverse = compute_avg_pairwise_cosine_distance(diverse, seed=42)
    passed = dist_diverse > dist_identical
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Diverse embeddings → dist > identical: "
          f"{dist_diverse:.4f} > {dist_identical:.6f}")

    # ── Test C: PCA serialization round-trip ──
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        pca_path = Path(tmpdir) / "test_pca.pkl"
        save_pca(pca, pca_path, corpus_hash="test_hash_abc")
        pca_loaded = load_pca(pca_path)

        # Projections must match
        proj_original = pca.transform(fake_embeddings[:10])
        proj_loaded   = pca_loaded.transform(fake_embeddings[:10])
        passed = np.allclose(proj_original, proj_loaded, atol=1e-5)
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] PCA serialization round-trip: "
              f"max_diff={np.abs(proj_original - proj_loaded).max():.2e}")

        # Sidecar JSON must exist
        passed = Path(pca_path).with_suffix(".json").exists()
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] PCA sidecar JSON created")

    # ── Test D: intrinsic dimensionality proxy gives reasonable values ──
    # Low-dim data embedded in high-dim space should give low ID
    low_dim_signal = rng.standard_normal((200, 3))
    random_proj = rng.standard_normal((3, 384))
    low_dim_embedded = (low_dim_signal @ random_proj).astype(np.float32)
    id_low = _pca_dimensionality_proxy(low_dim_embedded)

    high_dim_embedded = rng.standard_normal((200, 384)).astype(np.float32)
    id_high = _pca_dimensionality_proxy(high_dim_embedded)

    passed = id_low < id_high
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Low-dim data has lower ID than high-dim: "
          f"{id_low:.1f} < {id_high:.1f}")

    print()
    print("  ✓ SEMANTIC SELF-TEST PASSED" if all_passed
          else "  ✗ SEMANTIC SELF-TEST FAILED")
    return all_passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO semantic baseline computation")
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path(__file__).parent / "corpus")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "measurements")
    parser.add_argument("--model-id", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from phase1_baseline.corpus.ingest import load_corpus
    documents = load_corpus(args.corpus_dir)

    import json as _json
    manifest_file = args.corpus_dir / "manifest.json"
    corpus_hash = "unknown"
    if manifest_file.exists():
        with open(manifest_file) as f:
            corpus_hash = _json.load(f).get("corpus_sha256", "unknown")

    compute_semantic_baseline(
        documents, args.output_dir,
        model_id=args.model_id,
        n_pca_components=args.pca_components,
        corpus_hash=corpus_hash,
        seed=args.seed,
    )
