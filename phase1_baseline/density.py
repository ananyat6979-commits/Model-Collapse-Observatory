"""
MCO Phase 1 — Density Baseline (Tail Mass)
============================================
Fits the kernel density estimator on PCA-projected baseline embeddings.
Computes and freezes the tail threshold (5th percentile log-likelihood).

Produces:
    measurements/kde_params.pkl        — fitted KDE object (sklearn)
    measurements/tail_threshold.txt    — single float, the frozen threshold
    measurements/kde_params.json       — key params for verification without unpickling

The tail threshold is computed ONCE on Phase 1 baseline data and frozen permanently.
It never changes in any subsequent phase. Phase 3 tail_mass.py applies the baseline
KDE to generated embeddings and reports what fraction fall below this fixed threshold.

That fraction decreasing → collapse is occurring (model avoids rare phenomena).

Self-test:
    python phase1_baseline/density.py --self-test
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


# ── KDE fitting ───────────────────────────────────────────────────────────────

def fit_kde(
    embeddings_pca: np.ndarray,
    bandwidth_method: str = "cross_validation",
    cv_folds: int = 5,
    fit_sample_size: int = 2000,
    seed: int = 42,
) -> tuple[KernelDensity, float]:
    """
    Fit a Gaussian KDE on PCA-projected baseline embeddings.

    Args:
        embeddings_pca: array of shape (n_docs, n_pca_components)
        bandwidth_method: 'cross_validation' (slower, better) or 'scott' (fast)
        cv_folds: number of cross-validation folds for bandwidth selection
        fit_sample_size: subsample size for KDE fitting (CPU memory budget)
        seed: for reproducible CV splits

    Returns:
        kde: fitted KernelDensity
        bandwidth: selected bandwidth value (serialized separately)
    """
    rng = np.random.default_rng(seed)

    # Subsample if necessary
    n = len(embeddings_pca)
    if n > fit_sample_size:
        indices = rng.choice(n, size=fit_sample_size, replace=False)
        indices.sort()
        fit_data = embeddings_pca[indices]
        print(f"  KDE fit: subsampled {fit_sample_size} of {n} embeddings")
    else:
        fit_data = embeddings_pca
        print(f"  KDE fit: using all {n} embeddings")

    if bandwidth_method == "cross_validation":
        bandwidth = _select_bandwidth_cv(fit_data, cv_folds=cv_folds, seed=seed)
    elif bandwidth_method == "scott":
        n_d = fit_data.shape[0]
        d = fit_data.shape[1]
        bandwidth = float(n_d ** (-1 / (d + 4)))  # Scott's rule
        print(f"  KDE bandwidth (Scott's rule): {bandwidth:.6f}")
    else:
        raise ValueError(f"Unknown bandwidth_method: {bandwidth_method}")

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(fit_data)
    print(f"  KDE fitted with bandwidth={bandwidth:.6f}")
    return kde, bandwidth


def _select_bandwidth_cv(
    data: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> float:
    """
    Select KDE bandwidth via cross-validated log-likelihood maximization.

    Searches over a log-spaced grid of bandwidths. Seed is passed to
    GridSearchCV to ensure reproducible CV splits.

    Note: This is the expensive step (~5–10 minutes on CPU for 2k × 20d data).
    Run once, serialize the result, never re-run.
    """
    import warnings
    from sklearn.model_selection import KFold

    # Log-spaced bandwidth grid: 10 candidates from 0.01 to 2.0
    # Wide enough to include the optimal value for typical sentence embeddings
    bandwidths = np.logspace(-2, np.log10(2.0), 15)

    print(f"  Selecting KDE bandwidth via {cv_folds}-fold CV "
          f"over {len(bandwidths)} candidates...")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": bandwidths},
        cv=cv,
        n_jobs=1,          # deterministic — no parallel randomness
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress sklearn convergence noise
        grid.fit(data)

    bandwidth = float(grid.best_params_["bandwidth"])
    print(f"  Best bandwidth: {bandwidth:.6f} "
          f"(CV log-likelihood: {grid.best_score_:.4f})")
    return bandwidth


# ── Tail threshold computation ─────────────────────────────────────────────────

def compute_tail_threshold(
    kde: KernelDensity,
    eval_embeddings_pca: np.ndarray,
    percentile: float = 5.0,
) -> float:
    """
    Compute the tail threshold: Pth percentile of log-likelihood under the KDE,
    evaluated on HELD-OUT baseline embeddings (not the KDE training data).

    Critical: the eval data must be DIFFERENT from the KDE fit data.
    If you compute the threshold on the training data, the KDE overfits to
    its own training points (which score higher than held-out samples from
    the same distribution). This would make the threshold too high, causing
    far more than 5% of generated text to fall below it — giving a false
    collapse signal even for a healthy model.

    In build_corpus: use 60% of baseline embeddings to fit KDE, 40% to set threshold.

    Args:
        kde: fitted KDE
        eval_embeddings_pca: HELD-OUT baseline PCA embeddings (not used in KDE fit)
        percentile: the threshold percentile (default 5.0)

    Returns:
        threshold: single float log-likelihood value (frozen permanently)
    """
    log_likelihoods = kde.score_samples(eval_embeddings_pca)
    threshold = float(np.percentile(log_likelihoods, percentile))
    pct_below = (log_likelihoods < threshold).mean() * 100
    print(f"  Tail threshold ({percentile}th percentile on {len(eval_embeddings_pca)} "
          f"held-out embeddings): {threshold:.6f}")
    print(f"  Hold-out coverage check: {pct_below:.1f}% below threshold "
          f"(should be ~{percentile}%)")
    return threshold


# ── Serialization ──────────────────────────────────────────────────────────────

def save_kde(
    kde: KernelDensity,
    threshold: float,
    output_dir: Path,
    bandwidth: float,
    corpus_hash: str = "unknown",
    percentile: float = 5.0,
) -> None:
    """
    Save KDE, threshold, and metadata to output_dir.

    Files written:
        kde_params.pkl       — sklearn KernelDensity object
        kde_params.json      — key parameters (for verification without unpickling)
        tail_threshold.txt   — single float on one line
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pickle the KDE
    kde_path = output_dir / "kde_params.pkl"
    with open(kde_path, "wb") as f:
        pickle.dump(kde, f, protocol=4)

    # JSON sidecar — allows verifying params without loading the pickle
    kde_json = {
        "_corpus_hash": corpus_hash,
        "_computed_at": __import__("datetime").datetime.now().isoformat(),
        "_sklearn_version": _get_sklearn_version(),
        "kernel": kde.kernel,
        "bandwidth": bandwidth,
        "bandwidth_selection": "cross_validation_5fold",
        "tail_threshold_value": threshold,
        "tail_threshold_percentile": percentile,
        "note": (
            "tail_threshold_value is frozen permanently. "
            "Do NOT recompute in Phase 3. Load from tail_threshold.txt."
        ),
    }
    kde_json_path = output_dir / "kde_params.json"
    with open(kde_json_path, "w") as f:
        json.dump(kde_json, f, indent=2)

    # Tail threshold as a plain text file — single float, easy to load anywhere
    threshold_path = output_dir / "tail_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(f"{threshold:.10f}\n")

    print(f"  Written: {kde_path}")
    print(f"  Written: {kde_json_path}")
    print(f"  Written: {threshold_path}")


def load_kde(output_dir: Path) -> tuple[KernelDensity, float]:
    """
    Load the saved KDE and tail threshold.
    Verifies sklearn version and warns on mismatch.

    Returns:
        kde: loaded KernelDensity
        threshold: the frozen tail threshold float
    """
    kde_path = output_dir / "kde_params.pkl"
    threshold_path = output_dir / "tail_threshold.txt"

    if not kde_path.exists():
        raise FileNotFoundError(f"KDE not found: {kde_path}. Run density.py first.")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Tail threshold not found: {threshold_path}.")

    with open(kde_path, "rb") as f:
        kde = pickle.load(f)

    with open(threshold_path) as f:
        threshold = float(f.read().strip())

    # Version check
    json_path = output_dir / "kde_params.json"
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        saved_ver = meta.get("_sklearn_version", "unknown")
        current_ver = _get_sklearn_version()
        if saved_ver != current_ver:
            print(f"[WARN] KDE sklearn version mismatch: "
                  f"saved={saved_ver}, current={current_ver}. "
                  f"Scores may differ. Consider re-fitting KDE.")

    return kde, threshold


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "unknown"


# ── Main computation ───────────────────────────────────────────────────────────

def compute_density_baseline(
    embeddings_pca: np.ndarray,
    output_dir: Path,
    corpus_hash: str = "unknown",
    bandwidth_method: str = "cross_validation",
    tail_percentile: float = 5.0,
    fit_sample_size: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Fit KDE on baseline PCA embeddings and serialize all density artifacts.

    Uses a 60/40 train/eval split:
      - KDE is fit on 60% of embeddings
      - Tail threshold is computed on the held-out 40%

    This prevents KDE overfitting from biasing the threshold upward.
    The KDE scores its own training points higher than held-out points
    from the same distribution — so computing the threshold on training
    data would set it too high, causing false collapse signals in Phase 3.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n = len(embeddings_pca)
    print(f"Fitting density baseline on {n:,} PCA embeddings...")
    print(f"  PCA space: {embeddings_pca.shape[1]} dimensions")

    # Train / eval split
    idx = rng.permutation(n)
    split = int(n * 0.6)
    train_embeddings = embeddings_pca[np.sort(idx[:split])]
    eval_embeddings  = embeddings_pca[np.sort(idx[split:])]
    print(f"  KDE train: {len(train_embeddings):,} | threshold eval: {len(eval_embeddings):,}")

    kde, bandwidth = fit_kde(
        train_embeddings,
        bandwidth_method=bandwidth_method,
        fit_sample_size=min(fit_sample_size, len(train_embeddings)),
        seed=seed,
    )

    # Threshold on HELD-OUT eval set — unbiased generalization estimate
    threshold = compute_tail_threshold(kde, eval_embeddings, percentile=tail_percentile)

    save_kde(kde, threshold, output_dir, bandwidth, corpus_hash, tail_percentile)

    log_ll = kde.score_samples(eval_embeddings)
    actual_below = (log_ll < threshold).mean() * 100
    if abs(actual_below - tail_percentile) > 3.0:
        print(f"  [WARN] Tail coverage {actual_below:.1f}% deviates from "
              f"expected {tail_percentile:.1f}% on held-out set — KDE may be poorly fitted")

    return {
        "kde": kde,
        "bandwidth": bandwidth,
        "tail_threshold": threshold,
        "actual_tail_coverage_pct": round(float(actual_below), 3),
        "n_train": len(train_embeddings),
        "n_eval": len(eval_embeddings),
    }


# ── Self-test ──────────────────────────────────────────────────────────────────

def run_self_test() -> bool:
    """
    Verify KDE and tail threshold behavior on synthetic data.

    Key assertion: when we pass a test set from the same distribution,
    approximately 5% should fall below the 5th-percentile threshold.
    When we pass out-of-distribution data (collapsed = all at center),
    fewer than 5% should fall below (model avoids rare regions).
    """
    print("\n── Density Baseline Self-Test ────────────────────────────────────")
    all_passed = True
    rng = np.random.default_rng(42)

    # ── Test A & B: use explicit train/eval split ──
    rng2 = np.random.default_rng(99)
    baseline_all = rng2.standard_normal((800, 20)).astype(np.float32)
    train_data = baseline_all[:480]   # 60%
    eval_data  = baseline_all[480:]   # 40%

    kde, bandwidth = fit_kde(train_data, bandwidth_method="scott", fit_sample_size=480, seed=42)
    passed = bandwidth > 0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] KDE fitted, bandwidth={bandwidth:.6f}")

    threshold = compute_tail_threshold(kde, eval_data, percentile=5.0)

    # Test A: eval samples from same distribution → ~5% below threshold
    pct_below_eval = (kde.score_samples(eval_data) < threshold).mean() * 100
    passed = 1.0 <= pct_below_eval <= 15.0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Held-out baseline: {pct_below_eval:.1f}% below "
          f"threshold (expect ~5%, allow 1–15%)")

    # Test B: collapsed samples (tight cluster at center) → FEWER tail samples
    collapsed = rng.standard_normal((200, 20)).astype(np.float32) * 0.1
    pct_below_collapsed = (kde.score_samples(collapsed) < threshold).mean() * 100
    passed = pct_below_collapsed < pct_below_eval
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Collapsed < held-out tail: "
          f"{pct_below_collapsed:.1f}% < {pct_below_eval:.1f}%")

    # Test C: serialization round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        save_kde(kde, threshold, out, bandwidth, corpus_hash="test")
        kde_loaded, threshold_loaded = load_kde(out)

        passed = abs(threshold_loaded - threshold) < 1e-8
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] Threshold round-trip: "
              f"|diff|={abs(threshold_loaded - threshold):.2e}")

        # Scores must match after reload
        scores_orig   = kde.score_samples(train_data[:20])
        scores_loaded = kde_loaded.score_samples(train_data[:20])
        passed = np.allclose(scores_orig, scores_loaded, atol=1e-5)
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] KDE score round-trip: "
              f"max_diff={np.abs(scores_orig - scores_loaded).max():.2e}")

        # tail_threshold.txt must contain the float
        txt = (out / "tail_threshold.txt").read_text().strip()
        passed = abs(float(txt) - threshold) < 1e-8
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] tail_threshold.txt readable: "
              f"value={float(txt):.6f}")

    print()
    print("  ✓ DENSITY SELF-TEST PASSED" if all_passed
          else "  ✗ DENSITY SELF-TEST FAILED")
    return all_passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO density baseline computation")
    parser.add_argument("--embeddings-pca", type=Path,
                        help="Path to semantic_embeddings_pca.npy")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "measurements")
    parser.add_argument("--bandwidth-method", choices=["cross_validation", "scott"],
                        default="cross_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    if not args.embeddings_pca:
        print("Error: --embeddings-pca required unless --self-test")
        sys.exit(1)

    embeddings_pca = np.load(args.embeddings_pca, allow_pickle=False)
    compute_density_baseline(
        embeddings_pca=embeddings_pca,
        output_dir=args.output_dir,
        bandwidth_method=args.bandwidth_method,
        seed=args.seed,
    )
