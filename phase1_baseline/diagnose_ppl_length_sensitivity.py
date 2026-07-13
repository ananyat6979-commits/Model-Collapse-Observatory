"""
MCO — EXP-008: PPL vs. Document-Length Sensitivity Diagnostic
================================================================
Resolves the open Phase 1 question: why does mean_ppl = 44.4445
(phase1_baseline/measurements/ppl_baseline.json) fall outside the
documented "expected range: 15-30 for DistilGPT-2 on Wikipedia"
(phase1_baseline/perplexity.py, compute_ppl_baseline_stats)?

Confirmed facts (see references/experiment_log.md EXP-008 write-up):
  - Corpus documents are capped at 50-135 whitespace tokens
    (config.yaml: corpus_min/max_doc_length_tokens), calibrated to a
    256-subword-token encoder budget with margin. Short by
    article-length standards.
  - phase1_baseline/perplexity.py::compute_perplexity() uses a
    512-token sliding window (stride=256) that NEVER triggers for
    documents this short — every document is scored in a single,
    UNTRUNCATED forward pass. "Sliding window" in that docstring is
    misleading framing, not a bug.
  - The "15-30" expected range has no documented derivation anywhere
    in this repo. It is asserted, not sourced — likely a rough prior
    for full-length Wikipedia articles (hundreds-thousands of tokens),
    not 50-135 token snippets.
  - The measured distribution is right-skewed (mean 44.4 > median
    39.7, tail to 287.1), consistent with short-document PPL
    inflation, but this repo alone cannot prove causation.

What this script tests:
  H: mean per-document PPL, computed via the EXACT SAME untruncated
     method as the original baseline run, correlates positively with
     each document's actual token length. I.e., shorter documents in
     this corpus show higher PPL than longer ones, which would explain
     the 44.4 vs. "15-30 for full articles" gap as a corpus-composition
     effect (short-form snippets), not a bug in the PPL computation.

Critical methodological note — this REPLACES an earlier draft of this
diagnostic that reimplemented perplexity computation with an artificial
`truncation=True, max_length=X` cap. That version tested the wrong
variable: since corpus documents are already <=135 tokens, capping at
64/135/256 tokens rarely binds, and the earlier draft's tokenizer call
diverged from the original single-window, untruncated, no-cap method
in phase1_baseline/perplexity.py::compute_perplexity(). This version
imports and calls that function directly — zero reimplementation, zero
risk of a second, subtly different PPL definition drifting from the
production baseline.

Usage:
    python phase1_baseline/diagnose_ppl_length_sensitivity.py

Requires: the full corpus at phase1_baseline/corpus/documents.jsonl
(not included in the shared repo archive — pull from the Kaggle
dataset per phase2_simulation/phase2_README.md conventions, or run
against a smaller local corpus for a sanity check; sample size is
configurable below).

Runtime: CPU only. ~15-25 min for 5,000 documents on a modern laptop
(DistilGPT-2 is 82M params, single untruncated forward pass per doc).

Produces:
    phase1_baseline/measurements/ppl_length_sensitivity.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_baseline.perplexity import compute_perplexity, load_g0_model  # noqa: E402
from phase1_baseline.corpus.ingest import load_corpus  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
MODEL_ID = "distilgpt2"
SEED = 42
SAMPLE_SIZE = None  # None = full corpus (matches original 44.4445 run exactly).
                     # Set to e.g. 1000 for a fast local sanity check.
LENGTH_BUCKETS = [(0, 75), (75, 95), (95, 115), (115, 136)]
FULL_CORPUS_REFERENCE_MEAN_PPL = 44.4445  # from ppl_baseline.json, for cross-check


def bucket_label(n_tokens: int) -> str | None:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_tokens < hi:
            return f"{lo}-{hi - 1}"
    return None


def main():
    corpus_file = CORPUS_DIR / "documents.jsonl"
    manifest_file = CORPUS_DIR / "manifest.json"
    if not corpus_file.exists():
        print(f"ERROR: {corpus_file} not found.")
        print("Pull documents.jsonl from the Kaggle dataset first (see phase2_simulation/phase2_README.md")
        print("for the Kaggle access pattern used elsewhere in this project), then re-run.")
        return

    print("Loading corpus via phase1_baseline.corpus.ingest.load_corpus "
          "(verifies SHA-256 against manifest.json)...")
    documents = load_corpus(CORPUS_DIR)  # raises ValueError on hash mismatch — do not bypass this
    print(f"  {len(documents)} documents loaded, hash verified")

    with open(manifest_file) as f:
        corpus_hash = json.load(f).get("corpus_sha256", "unknown")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(documents):
        rng = np.random.default_rng(SEED)
        idx = sorted(rng.choice(len(documents), SAMPLE_SIZE, replace=False).tolist())
        documents = [documents[i] for i in idx]
        print(f"  Using {len(documents)}-document subsample (seed={SEED}). "
              f"NOTE: this will NOT exactly reproduce 44.4445 (that used all 5000).")
    else:
        print(f"  Using full corpus ({len(documents)} docs) — directly comparable to "
              f"the original {FULL_CORPUS_REFERENCE_MEAN_PPL} baseline.")

    print(f"\nLoading {MODEL_ID} (same load path as the original baseline run)...")
    model, tokenizer = load_g0_model(MODEL_ID, device="cpu")

    print(f"\nComputing per-document PPL via the PRODUCTION method "
          f"(compute_perplexity, untruncated, sliding-window-capable)...")
    t0 = time.time()
    ppls = compute_perplexity(model, tokenizer, documents, device="cpu")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    print("\nComputing per-document token lengths (whitespace-split, matching "
          "corpus_min/max_doc_length_tokens semantics in config.yaml)...")
    lengths = [len(doc.split()) for doc in documents]

    # ── Pair up, drop invalid PPLs ──────────────────────────────────────
    paired = [(ln, p) for ln, p in zip(lengths, ppls)
              if p is not None and np.isfinite(p)]
    n_invalid = len(ppls) - len(paired)
    lengths_clean = [ln for ln, _ in paired]
    ppls_clean = [p for _, p in paired]

    print(f"\n  n_valid={len(paired)}  n_invalid={n_invalid}")
    print(f"  Full-sample mean PPL: {np.mean(ppls_clean):.4f} "
          f"(reference baseline: {FULL_CORPUS_REFERENCE_MEAN_PPL})")

    # ── Correlation: does PPL track length? ─────────────────────────────
    from scipy import stats as scipy_stats
    corr, p_val = scipy_stats.spearmanr(lengths_clean, ppls_clean)
    print(f"\n  Spearman(length, ppl): rho={corr:.4f}  p={p_val:.6f}  n={len(paired)}")

    # ── Bucketed breakdown ────────────────────────────────────────────
    buckets: dict[str, list[float]] = {}
    for ln, p in paired:
        label = bucket_label(ln)
        if label is None:
            continue
        buckets.setdefault(label, []).append(p)

    bucket_stats = {}
    print(f"\n  {'Length bucket':>15} | {'n':>5} | {'mean_ppl':>10} | {'median_ppl':>12}")
    print("  " + "-" * 52)
    for lo, hi in LENGTH_BUCKETS:
        label = f"{lo}-{hi - 1}"
        vals = buckets.get(label, [])
        if not vals:
            continue
        bucket_stats[label] = {
            "n": len(vals),
            "mean_ppl": round(float(np.mean(vals)), 4),
            "median_ppl": round(float(np.median(vals)), 4),
            "std_ppl": round(float(np.std(vals)), 4),
        }
        print(f"  {label:>15} | {len(vals):>5} | {bucket_stats[label]['mean_ppl']:>10.2f} "
              f"| {bucket_stats[label]['median_ppl']:>12.2f}")

    bucket_means = [bucket_stats[f"{lo}-{hi-1}"]["mean_ppl"]
                    for lo, hi in LENGTH_BUCKETS
                    if f"{lo}-{hi - 1}" in bucket_stats]
    monotonic_decreasing = all(
        bucket_means[i] >= bucket_means[i + 1] for i in range(len(bucket_means) - 1)
    )

    # ── Verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    significant_negative = (p_val < 0.01) and (corr < 0)
    if significant_negative:
        print("HYPOTHESIS SUPPORTED: PPL decreases significantly with document")
        print("length (rho=%.3f, p=%.2e). Short-document composition plausibly" % (corr, p_val))
        print("explains the 44.4 vs '15-30 (full articles)' gap. This is a")
        print("corpus-composition property, not a computation bug, and does NOT")
        print("bias G0-vs-Gk relative comparisons since every generation is")
        print("measured against documents of the same length distribution.")
    else:
        print("HYPOTHESIS NOT SUPPORTED (or not significant): no strong negative")
        print("length-PPL correlation found. Length-sensitivity does not explain")
        print("the gap on its own. Next candidates to check, in order of cost:")
        print("  1. Manually read 10-20 sample documents for content/preprocessing")
        print("     artifacts (infobox fragments, disambiguation stubs, etc.)")
        print("  2. Confirm BOS/EOS token handling matches standard DistilGPT-2")
        print("     PPL evaluation convention.")
        print("  3. Verify model_weights_sha256 in config.yaml against the actual")
        print("     loaded distilgpt2 checkpoint (rule out a checkpoint mismatch).")
    print("=" * 65)

    # ── Save ─────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "measurements" / "ppl_length_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment_id": "EXP-008",
            "purpose": "Resolve Phase 1 PPL-gate gap (44.4445 vs documented 15-30 "
                       "expected range) — test document-length sensitivity using the "
                       "PRODUCTION compute_perplexity() method, not a reimplementation.",
            "corpus_hash": corpus_hash,
            "model_id": MODEL_ID,
            "seed": SEED,
            "n_documents_tested": len(documents),
            "is_full_corpus": SAMPLE_SIZE is None,
            "n_valid": len(paired),
            "n_invalid": n_invalid,
            "full_sample_mean_ppl": round(float(np.mean(ppls_clean)), 4),
            "full_sample_median_ppl": round(float(np.median(ppls_clean)), 4),
            "reference_baseline_mean_ppl": FULL_CORPUS_REFERENCE_MEAN_PPL,
            "spearman_length_ppl": {"rho": round(float(corr), 4), "p_value": float(p_val)},
            "length_buckets": bucket_stats,
            "bucket_means_monotonic_decreasing": monotonic_decreasing,
            "hypothesis_supported": bool(significant_negative),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Next: log this result as EXP-008 in references/experiment_log.md "
          "and update THREATS_TO_VALIDITY.md with the resolution.")


if __name__ == "__main__":
    main()