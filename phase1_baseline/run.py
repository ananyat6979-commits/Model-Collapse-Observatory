#!/usr/bin/env python3
"""
MCO Phase 1 — Main Run Script
==============================
Executes the complete Phase 1 baseline pipeline in order.

Full run (requires Wikipedia dump):
    python phase1_baseline/run.py --seed 42 --dump /path/to/enwiki-*.xml.bz2

Reproducibility verification only (requires completed pipeline):
    python phase1_baseline/run.py --seed 42 --test-only

Resume an interrupted run (skip stages whose outputs exist):
    python phase1_baseline/run.py --seed 42 --dump /path/to/dump.xml.bz2 --skip-existing

Reproducibility contract:
    Two consecutive full runs with --seed 42 on the same dump must produce
    a reference_pack.pkl whose SHA-256 hash is identical.
    Verify with: python phase1_baseline/run.py --test-only (run twice, compare hashes)

Pipeline order (matches phase_criteria.md):
    Stage 1: Corpus ingestion → documents.jsonl + manifest.json
             (includes Stage 0: tokenizer length calibration)
    Stage 2: Lexical baseline → lexical_baseline.json, zipf_params.json
    Stage 3: Semantic baseline → semantic_baseline.json, pca_transform.pkl,
             semantic_embeddings.npy, semantic_embeddings_pca.npy (ALL docs)
    Stage 4: Density baseline → kde_params.pkl, tail_threshold.txt
    Stage 5: PPL baseline → ppl_baseline.json
    Stage 6: Pack assembly → reference_pack.pkl + reference_pack_manifest.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


# ── Reproducibility setup ─────────────────────────────────────────────────────

def setup_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"]         = "1"
    os.environ["MKL_NUM_THREADS"]         = "1"
    os.environ["OPENBLAS_NUM_THREADS"]    = "1"

    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        print("[WARN] torch not available — skipping torch seed setup")


# ── SHA-256 helpers ────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Reproducibility test ──────────────────────────────────────────────────────

def run_reproducibility_test(output_dir: Path) -> bool:
    print("\n── Reproducibility Test ──────────────────────────────────────────")

    checks = {
        "reference_pack.pkl":
            output_dir / "reference_pack.pkl",
        "reference_pack_manifest.json":
            output_dir / "reference_pack_manifest.json",
        "measurements/lexical_baseline.json":
            output_dir / "measurements" / "lexical_baseline.json",
        "measurements/zipf_params.json":
            output_dir / "measurements" / "zipf_params.json",
        "measurements/semantic_baseline.json":
            output_dir / "measurements" / "semantic_baseline.json",
        "measurements/pca_transform.pkl":
            output_dir / "measurements" / "pca_transform.pkl",
        "measurements/semantic_embeddings_pca.npy":
            output_dir / "measurements" / "semantic_embeddings_pca.npy",
        "measurements/kde_params.pkl":
            output_dir / "measurements" / "kde_params.pkl",
        "measurements/tail_threshold.txt":
            output_dir / "measurements" / "tail_threshold.txt",
        "measurements/ppl_baseline.json":
            output_dir / "measurements" / "ppl_baseline.json",
        "corpus/manifest.json":
            output_dir / "corpus" / "manifest.json",
        "corpus/documents.jsonl":
            output_dir / "corpus" / "documents.jsonl",
    }

    all_passed = True
    for name, path in checks.items():
        if not path.exists():
            print(f"  [FAIL] Missing: {name}")
            all_passed = False
        else:
            h = sha256_json(path) if path.suffix == ".json" else sha256_file(path)
            print(f"  [OK]   {name} — {h[:16]}...")

    print()
    if all_passed:
        from phase1_baseline.pack import verify_reference_pack
        all_passed = verify_reference_pack(output_dir)
    else:
        print("  ✗ REPRODUCIBILITY TEST FAILED — missing artifacts")
        print("    Run the full pipeline first.")

    return all_passed


# ── Baseline sanity checks ────────────────────────────────────────────────────

def run_sanity_checks(output_dir: Path) -> bool:
    """
    Validate baseline statistics are within expected ranges.
    Thresholds from phase_criteria.md.
    """
    print("\n── Baseline Sanity Checks ────────────────────────────────────────")
    all_passed = True
    m = output_dir / "measurements"

    def load_json(fname):
        p = m / fname
        return json.loads(p.read_text()) if p.exists() else None

    zipf     = load_json("zipf_params.json")
    lexical  = load_json("lexical_baseline.json")
    semantic = load_json("semantic_baseline.json")
    ppl      = load_json("ppl_baseline.json")

    checks = [
        ("Zipf alpha 0.8–1.2",
         zipf and 0.8 <= zipf.get("zipf_alpha", 0) <= 1.2,
         f"alpha={zipf.get('zipf_alpha', 'N/A') if zipf else 'MISSING'}"),

        ("Corpus TTR > 0.10",
         lexical and lexical.get("corpus_ttr", 0) > 0.10,
         f"TTR={lexical.get('corpus_ttr', 'N/A') if lexical else 'MISSING'}"),

        ("Intrinsic dimensionality > 10",
         semantic and semantic.get("intrinsic_dimensionality", 0) > 10,
         f"dim={semantic.get('intrinsic_dimensionality', 'N/A') if semantic else 'MISSING'}"),

        ("Mean PPL 15–60 (DistilGPT-2 on short Wikipedia docs)",
         ppl and 15 <= ppl.get("mean_ppl", 0) <= 60,
         f"mean_ppl={ppl.get('mean_ppl', 'N/A') if ppl else 'MISSING'}"),

        ("G0 weight hash recorded",
         ppl and bool(ppl.get("_model_weights_sha256")),
         (ppl.get("_model_weights_sha256", "MISSING")[:16] + "..."
          if ppl else "MISSING")),
    ]

    for name, result, detail in checks:
        print(f"  [{'OK  ' if result else 'FAIL'}] {name} — {detail}")
        if not result:
            all_passed = False

    print()
    print("  ✓ ALL SANITY CHECKS PASSED" if all_passed
          else "  ✗ SANITY CHECKS FAILED — investigate before Phase 2")
    return all_passed


# ── Pipeline stages ────────────────────────────────────────────────────────────

def _load_corpus_hash(output_dir: Path) -> str:
    manifest = output_dir / "corpus" / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text()).get("corpus_sha256", "unknown")
    return "unknown"


def stage_corpus(dump_path, output_dir: Path, seed: int,
                 skip_existing: bool) -> str:
    """
    Stage 1/6: Corpus ingestion.

    If skip_existing=True and a valid corpus already exists (manifest.json
    present with matching hash), returns the existing corpus hash immediately
    without opening the dump file. This allows resuming the pipeline after
    building the corpus via ingest_hf.py, without requiring the bz2 dump.
    """
    corpus_dir = output_dir / "corpus"
    manifest   = corpus_dir / "manifest.json"

    if skip_existing and manifest.exists():
        # Verify hash before trusting the existing corpus
        try:
            import hashlib
            corpus_file = corpus_dir / "documents.jsonl"
            with open(manifest) as f:
                import json as _json
                m = _json.load(f)
            expected = m.get("corpus_sha256", "")
            if expected and corpus_file.exists():
                h = hashlib.sha256()
                with open(corpus_file, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                actual = h.hexdigest()
                if actual == expected:
                    print(f"  [SKIP] Corpus exists and hash verified — {actual[:16]}...")
                    return actual
                else:
                    print(f"  [WARN] Corpus hash mismatch — rebuilding")
        except Exception as e:
            print(f"  [WARN] Could not verify existing corpus ({e}) — rebuilding")

    # Corpus doesn't exist or is invalid — must build it.
    if dump_path is None:
        raise RuntimeError(
            "Corpus not found and --dump was not provided.\n"
            "Options:\n"
            "  a) Build corpus via HF backend (recommended on Windows):\n"
            "       python phase1_baseline/corpus/ingest_hf.py "
            "--output-dir phase1_baseline/corpus\n"
            "     Then re-run with --skip-existing (no --dump needed).\n"
            "  b) Provide the bz2 dump path:\n"
            "       --dump /path/to/enwiki-*.xml.bz2\n"
        )
    if not Path(dump_path).exists():
        raise RuntimeError(f"Dump file not found: {dump_path}")

    print("  Running corpus ingestion (bz2 path)...")
    from phase1_baseline.corpus.ingest import build_corpus
    result = build_corpus(dump_path=Path(dump_path), output_dir=corpus_dir, seed=seed)
    return result["manifest"]["corpus_sha256"]


def stage_lexical(output_dir: Path, corpus_hash: str, skip_existing: bool) -> None:
    if skip_existing and (output_dir / "measurements" / "lexical_baseline.json").exists():
        print("  [SKIP] Lexical baseline exists")
        return
    print("  Computing lexical baseline...")
    from phase1_baseline.corpus.ingest import load_corpus
    from phase1_baseline.lexical_baseline import compute_lexical_baseline
    docs = load_corpus(output_dir / "corpus")
    compute_lexical_baseline(docs, output_dir / "measurements", corpus_hash=corpus_hash)


def stage_semantic(output_dir: Path, corpus_hash: str, seed: int,
                   skip_existing: bool) -> None:
    if skip_existing and (output_dir / "measurements" / "semantic_baseline.json").exists():
        print("  [SKIP] Semantic baseline exists")
        return
    print("  Computing semantic baseline (~10 min on CPU)...")
    from phase1_baseline.corpus.ingest import load_corpus
    from phase1_baseline.embeddings import compute_semantic_baseline
    docs = load_corpus(output_dir / "corpus")
    compute_semantic_baseline(
        docs, output_dir / "measurements",
        corpus_hash=corpus_hash, seed=seed,
    )


def stage_density(output_dir: Path, corpus_hash: str, seed: int,
                  skip_existing: bool) -> None:
    if skip_existing and (output_dir / "measurements" / "kde_params.pkl").exists():
        print("  [SKIP] Density baseline exists")
        return
    print("  Fitting KDE + tail threshold (CV may take ~10 min)...")
    import numpy as np
    from phase1_baseline.density import compute_density_baseline
    pca_path = output_dir / "measurements" / "semantic_embeddings_pca.npy"
    if not pca_path.exists():
        raise FileNotFoundError("semantic_embeddings_pca.npy missing — run Stage 3 first.")
    compute_density_baseline(
        embeddings_pca=np.load(pca_path, allow_pickle=False),
        output_dir=output_dir / "measurements",
        corpus_hash=corpus_hash, seed=seed,
    )


def stage_ppl(output_dir: Path, corpus_hash: str, seed: int,
              skip_existing: bool) -> None:
    if skip_existing and (output_dir / "measurements" / "ppl_baseline.json").exists():
        print("  [SKIP] PPL baseline exists")
        return
    print("  Computing PPL baseline under G0...")
    from phase1_baseline.corpus.ingest import load_corpus
    from phase1_baseline.perplexity import compute_perplexity_baseline
    docs = load_corpus(output_dir / "corpus")
    compute_perplexity_baseline(
        documents=docs, output_dir=output_dir / "measurements",
        corpus_hash=corpus_hash, seed=seed,
    )


def stage_pack(output_dir: Path, skip_existing: bool) -> None:
    if skip_existing and (output_dir / "reference_pack.pkl").exists():
        print("  [SKIP] Reference pack exists")
        return
    print("  Assembling reference pack...")
    from phase1_baseline.pack import build_reference_pack
    build_reference_pack(output_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MCO Phase 1 baseline pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump", type=Path, default=None,
                        help="Wikipedia XML dump (.xml or .xml.bz2). Required unless --test-only.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--test-only", action="store_true",
                        help="Verify completed pipeline. No execution.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip stages whose outputs already exist.")
    args = parser.parse_args()

    setup_determinism(args.seed)

    # Ensure project root is importable
    sys.path.insert(0, str(args.output_dir.parent))

    print(f"MCO Phase 1 — seed={args.seed}")
    print(f"Output: {args.output_dir.resolve()}")

    if args.test_only:
        ok = run_reproducibility_test(args.output_dir)
        ok = run_sanity_checks(args.output_dir) and ok
        if ok:
            print("\n✓ Phase 1 COMPLETE — safe to begin Phase 2.")
        else:
            print("\n✗ Phase 1 incomplete or failing checks.")
        sys.exit(0 if ok else 1)

    # --dump is required for a fresh corpus build.
    # With --skip-existing and a valid pre-built corpus (e.g. from ingest_hf.py),
    # --dump becomes optional — stage_corpus will verify and skip ingestion.
    corpus_dir    = args.output_dir / "corpus"
    corpus_exists = (corpus_dir / "manifest.json").exists() and \
                    (corpus_dir / "documents.jsonl").exists()

    if args.skip_existing and corpus_exists:
        # Corpus exists — dump not needed, stage_corpus will verify and skip
        if args.dump is not None and not args.dump.exists():
            print(f"[WARN] --dump path not found ({args.dump}) but corpus exists — ignoring dump")
            args.dump = None
        print("Corpus pre-built — dump not required for this run.\n" + "=" * 60)
    else:
        # Need to build corpus — dump is required
        if args.dump is None:
            print(
                "Error: --dump is required when corpus does not already exist.\n\n"
                "On Windows with limited RAM, build the corpus first using the\n"
                "HuggingFace streaming backend (no bz2 parsing, no MemoryError):\n\n"
                "  python phase1_baseline\\corpus\\ingest_hf.py "
                "--output-dir phase1_baseline\\corpus\n\n"
                "Then re-run with --skip-existing (--dump is optional at that point)."
            )
            sys.exit(1)
        if not args.dump.exists():
            print(f"Error: dump not found: {args.dump}")
            sys.exit(1)
        print(f"Dump: {args.dump}\n" + "=" * 60)

    print("\n[Stage 1/6] Corpus ingestion")
    corpus_hash = stage_corpus(args.dump, args.output_dir, args.seed, args.skip_existing)

    print("\n[Stage 2/6] Lexical baseline")
    stage_lexical(args.output_dir, corpus_hash, args.skip_existing)

    print("\n[Stage 3/6] Semantic baseline")
    stage_semantic(args.output_dir, corpus_hash, args.seed, args.skip_existing)

    print("\n[Stage 4/6] Density baseline")
    stage_density(args.output_dir, corpus_hash, args.seed, args.skip_existing)

    print("\n[Stage 5/6] PPL baseline")
    stage_ppl(args.output_dir, corpus_hash, args.seed, args.skip_existing)

    print("\n[Stage 6/6] Reference pack")
    stage_pack(args.output_dir, args.skip_existing)

    print("\n" + "=" * 60)
    ok = run_reproducibility_test(args.output_dir)
    ok = run_sanity_checks(args.output_dir) and ok

    if ok:
        print("\n✓ Phase 1 COMPLETE")
        print("  Next: mco_complete_phase in the MCP server, then Phase 2 on Kaggle.")
    else:
        print("\n✗ Pipeline finished but verification failed. Review output above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
