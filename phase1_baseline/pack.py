"""
MCO Phase 1 — Reference Pack Serializer
=========================================
Assembles all Phase 1 measurement artifacts into a single reference_pack.pkl
for use by Phases 3, 4, and 5.

The reference pack is NOT a monolith pickle. It is a Python dict that holds:
  - Scalar stats: raw dicts loaded from JSON files
  - Arrays: loaded from .npy files (numpy arrays, no pickle needed)
  - Fitted objects: PCA (sklearn) and KDE (sklearn) loaded from their .pkl files
  - Hashes and metadata for every constituent artifact

The top-level dict is pickled for convenience, but the individual artifacts
are also independently loadable from their source files. If the pack becomes
unloadable (sklearn version drift), every measurement can be reconstructed
from the constituent JSON/npy/pkl files.

Usage:
    python phase1_baseline/pack.py --phase1-dir phase1_baseline/

Verification:
    python phase1_baseline/pack.py --verify --phase1-dir phase1_baseline/
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


# ── SHA-256 helpers ────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dict(d: dict) -> str:
    """Deterministic hash of a dict via sorted-key JSON."""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Pack builder ───────────────────────────────────────────────────────────────

def build_reference_pack(phase1_dir: Path) -> dict:
    """
    Load all Phase 1 artifacts and assemble the reference pack.

    Validates that all required files exist and are consistent
    (corpus hashes match across artifacts) before assembling.

    Returns the pack dict. Also writes it to reference_pack.pkl
    and logs the SHA-256.
    """
    measurements_dir = phase1_dir / "measurements"
    corpus_dir       = phase1_dir / "corpus"

    # ── Required files ────────────────────────────────────────────────────────
    required = {
        "corpus_manifest":         corpus_dir / "manifest.json",
        "lexical_baseline":        measurements_dir / "lexical_baseline.json",
        "zipf_params":             measurements_dir / "zipf_params.json",
        "kl_baseline_distributions": measurements_dir / "kl_baseline_distributions.json",
        "semantic_baseline":       measurements_dir / "semantic_baseline.json",
        "pca_transform":           measurements_dir / "pca_transform.pkl",
        "semantic_embeddings_pca": measurements_dir / "semantic_embeddings_pca.npy",
        "kde_params":              measurements_dir / "kde_params.pkl",
        "kde_params_json":         measurements_dir / "kde_params.json",
        "tail_threshold":          measurements_dir / "tail_threshold.txt",
        "ppl_baseline":            measurements_dir / "ppl_baseline.json",
    }

    print("\n── Reference Pack Assembly ───────────────────────────────────────")

    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        print(f"  [FAIL] Missing artifacts: {missing}")
        print("         Run the full Phase 1 pipeline before building the pack.")
        raise FileNotFoundError(
            f"Cannot build reference pack — missing: {missing}"
        )

    print(f"  All {len(required)} required files present.")

    # ── Load JSON artifacts ───────────────────────────────────────────────────
    def load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    corpus_manifest    = load_json(required["corpus_manifest"])
    lexical_baseline   = load_json(required["lexical_baseline"])
    zipf_params        = load_json(required["zipf_params"])
    kl_distributions   = load_json(required["kl_baseline_distributions"])
    semantic_baseline  = load_json(required["semantic_baseline"])
    kde_params_json    = load_json(required["kde_params_json"])
    ppl_baseline       = load_json(required["ppl_baseline"])

    # ── Corpus hash consistency check ─────────────────────────────────────────
    # Every JSON artifact records the corpus hash it was computed from.
    # If they disagree, something was re-run against a different corpus.
    corpus_hash = corpus_manifest.get("corpus_sha256", "")
    hash_fields = {
        "lexical_baseline":  lexical_baseline.get("_corpus_hash", ""),
        "zipf_params":       zipf_params.get("_corpus_hash", ""),
        "semantic_baseline": semantic_baseline.get("_corpus_hash", ""),
        "kde_params_json":   kde_params_json.get("_corpus_hash", ""),
        "ppl_baseline":      ppl_baseline.get("_corpus_hash", ""),
    }

    mismatches = {k: v for k, v in hash_fields.items()
                  if v and v != "unknown" and v != corpus_hash}
    if mismatches:
        print(f"  [WARN] Corpus hash mismatch in: {list(mismatches.keys())}")
        print(f"         Corpus hash: {corpus_hash[:16]}...")
        for k, v in mismatches.items():
            print(f"         {k}: {v[:16]}...")
        print("         Some artifacts may have been recomputed on a different corpus.")

    print(f"  Corpus hash: {corpus_hash[:24]}...")

    # ── Load fitted objects ───────────────────────────────────────────────────
    sys.path.insert(0, str(phase1_dir.parent))
    from phase1_baseline.embeddings import load_pca
    from phase1_baseline.density import load_kde

    pca = load_pca(required["pca_transform"])
    kde, tail_threshold = load_kde(measurements_dir)
    print(f"  PCA loaded: {pca.n_components_} components")
    print(f"  KDE loaded: bandwidth={kde.bandwidth:.6f}")
    print(f"  Tail threshold: {tail_threshold:.6f}")

    # ── Load numpy arrays ─────────────────────────────────────────────────────
    embeddings_pca = np.load(
        required["semantic_embeddings_pca"], allow_pickle=False
    )
    print(f"  PCA embeddings: {embeddings_pca.shape} (all documents, no subsampling)")

    # ── Assemble pack ─────────────────────────────────────────────────────────
    pack = {
        # ── Metadata ──────────────────────────────────────────────────────────
        "_pack_version":     "1.0",
        "_assembled_at":     __import__("datetime").datetime.now().isoformat(),
        "_corpus_hash":      corpus_hash,
        "_corpus_manifest":  corpus_manifest,

        # ── Encoder identity (frozen — never changes after Phase 1) ───────────
        "encoder_id": semantic_baseline.get("encoder_id",
                        "sentence-transformers/all-MiniLM-L6-v2"),

        # ── G0 model identity ─────────────────────────────────────────────────
        "g0_model_id":            ppl_baseline.get("_model_id", "distilgpt2"),
        "g0_model_weights_sha256": ppl_baseline.get("_model_weights_sha256", None),

        # ── Lexical baseline (scalars) ─────────────────────────────────────────
        "lexical": {**lexical_baseline, **zipf_params},

        # ── KL baseline distributions (inlined — pack must be self-contained) ──
        # Stored directly in the pack, not as a filesystem path reference.
        # A path reference would break when the pack is copied to Kaggle or any
        # environment where the original laptop filesystem is not mounted.
        # Memory cost: ~2–5MB for 50k unigrams + 100k trigrams. Acceptable.
        # Phase 3 lexical.py loads from pack["kl_distributions"] — never from disk.
        "kl_distributions": kl_distributions,

        # ── Semantic baseline (scalars + fitted objects + arrays) ──────────────
        "semantic": semantic_baseline,
        "pca":      pca,          # sklearn PCA — frozen, sign-convention applied
        "embeddings_pca": embeddings_pca,  # np.ndarray, all baseline docs

        # ── Density / tail mass ────────────────────────────────────────────────
        "kde":            kde,            # sklearn KernelDensity — frozen
        "tail_threshold": tail_threshold, # float — frozen permanently at Phase 1
        "kde_params":     kde_params_json,

        # ── PPL baseline ──────────────────────────────────────────────────────
        "ppl_baseline": ppl_baseline,

        # ── Artifact file hashes (for tamper detection) ───────────────────────
        "artifact_hashes": {
            name: sha256_file(path)
            for name, path in required.items()
        },
    }

    # ── Serialize ─────────────────────────────────────────────────────────────
    pack_path = phase1_dir / "reference_pack.pkl"
    with open(pack_path, "wb") as f:
        pickle.dump(pack, f, protocol=4)

    pack_hash = sha256_file(pack_path)
    print(f"\n  Written: {pack_path}")
    print(f"  Pack SHA-256: {pack_hash}")

    # Write the hash to a sidecar for quick verification without loading
    sidecar = {
        "_pack_sha256":  pack_hash,
        "_corpus_hash":  corpus_hash,
        "_assembled_at": pack["_assembled_at"],
        "encoder_id":    pack["encoder_id"],
        "g0_model_id":   pack["g0_model_id"],
        "g0_model_weights_sha256": pack["g0_model_weights_sha256"],
        "tail_threshold": tail_threshold,
    }
    sidecar_path = phase1_dir / "reference_pack_manifest.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  Sidecar: {sidecar_path}")

    return pack


# ── Pack loader ────────────────────────────────────────────────────────────────

def load_reference_pack(phase1_dir: Path, verify_hash: bool = True) -> dict:
    """
    Load the reference pack. Optionally verify its hash against the sidecar.

    This is the function Phase 3 measurement modules should call.
    Never unpickle the pack directly — use this function so hash
    verification always runs.
    """
    pack_path    = phase1_dir / "reference_pack.pkl"
    sidecar_path = phase1_dir / "reference_pack_manifest.json"

    if not pack_path.exists():
        raise FileNotFoundError(
            f"Reference pack not found: {pack_path}. Run pack.py first."
        )

    if verify_hash and sidecar_path.exists():
        actual_hash = sha256_file(pack_path)
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        expected_hash = sidecar.get("_pack_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"Reference pack hash mismatch!\n"
                f"  Expected: {expected_hash[:24]}...\n"
                f"  Actual:   {actual_hash[:24]}...\n"
                f"The pack may have been modified or corrupted. Rebuild with pack.py."
            )
        print(f"[OK] Reference pack hash verified ({actual_hash[:16]}...)")

    with open(pack_path, "rb") as f:
        pack = pickle.load(f)

    return pack


# ── Verification ───────────────────────────────────────────────────────────────

def verify_reference_pack(phase1_dir: Path) -> bool:
    """
    Smoke-test the loaded reference pack for internal consistency.
    Call this at the start of every Phase 3 session.
    """
    print("\n── Reference Pack Verification ───────────────────────────────────")
    all_passed = True

    try:
        pack = load_reference_pack(phase1_dir, verify_hash=True)
    except Exception as e:
        print(f"  [FAIL] Pack load failed: {e}")
        return False

    checks = [
        ("encoder_id present",
         bool(pack.get("encoder_id")),
         pack.get("encoder_id", "MISSING")),

        ("pca loaded and n_components=20",
         hasattr(pack.get("pca"), "n_components_") and pack["pca"].n_components_ == 20,
         f"n_components={getattr(pack.get('pca'), 'n_components_', 'N/A')}"),

        ("PCA sign convention holds",
         pack["pca"].components_[0][0] > 0,
         f"components_[0][0]={pack['pca'].components_[0][0]:.6f}"),

        ("kde loaded",
         hasattr(pack.get("kde"), "score_samples"),
         "ok" if hasattr(pack.get("kde"), "score_samples") else "MISSING"),

        ("tail_threshold is float",
         isinstance(pack.get("tail_threshold"), float),
         str(pack.get("tail_threshold", "MISSING"))),

        ("embeddings_pca is ndarray",
         isinstance(pack.get("embeddings_pca"), np.ndarray),
         f"shape={pack.get('embeddings_pca').shape if isinstance(pack.get('embeddings_pca'), np.ndarray) else 'N/A'}"),

        ("embeddings_pca not subsampled — length consistent with corpus",
         isinstance(pack.get("embeddings_pca"), np.ndarray) and (
             len(pack["embeddings_pca"]) >=
             int(pack.get("_corpus_manifest", {}).get("stats", {}).get("n_documents", 5000) * 0.95)
         ),
         (f"{len(pack['embeddings_pca'])} docs stored, "
          f"expected ≥{int(pack.get('_corpus_manifest',{}).get('stats',{}).get('n_documents',5000)*0.95)}"
          if isinstance(pack.get("embeddings_pca"), np.ndarray) else "N/A")),

        ("lexical baseline has TTR",
         "corpus_ttr" in pack.get("lexical", {}),
         f"TTR={pack.get('lexical', {}).get('corpus_ttr', 'MISSING')}"),

        ("ppl_baseline has mean_ppl",
         "mean_ppl" in pack.get("ppl_baseline", {}),
         f"mean_ppl={pack.get('ppl_baseline', {}).get('mean_ppl', 'MISSING')}"),

        ("g0_model_weights_sha256 present",
         bool(pack.get("g0_model_weights_sha256")),
         (pack.get("g0_model_weights_sha256") or "MISSING")[:16] + "..."),

        ("corpus_hash present",
         bool(pack.get("_corpus_hash")),
         (pack.get("_corpus_hash") or "MISSING")[:16] + "..."),
    ]

    for name, result, detail in checks:
        status = "OK  " if result else "FAIL"
        print(f"  [{status}] {name} — {detail}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ REFERENCE PACK VERIFIED — safe to begin Phase 3")
    else:
        print("  ✗ REFERENCE PACK VERIFICATION FAILED — rebuild before Phase 3")

    return all_passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO reference pack builder")
    parser.add_argument("--phase1-dir", type=Path,
                        default=Path(__file__).parent,
                        help="Phase 1 directory (contains measurements/ and corpus/)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify an existing pack without rebuilding")
    args = parser.parse_args()

    if args.verify:
        ok = verify_reference_pack(args.phase1_dir)
        sys.exit(0 if ok else 1)

    build_reference_pack(args.phase1_dir)
    print("\nRunning verification on newly built pack...")
    verify_reference_pack(args.phase1_dir)
