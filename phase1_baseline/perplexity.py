"""
MCO Phase 1 — Perplexity Baseline
====================================
Computes per-document perplexity under the pretrained G0 DistilGPT-2 checkpoint.

This establishes the reference PPL distribution against which Phase 3's
perplexity inversion measurements are anchored.

Critical: The G0 checkpoint used here MUST be saved to the Kaggle Dataset
BEFORE any Phase 2 fine-tuning begins. If G0 is lost, perplexity inversion
loses its reference anchor and all subsequent measurements are incomparable.

Produces:
    measurements/ppl_baseline.json  — mean, std, percentiles, histogram

Self-test (no real model needed):
    python phase1_baseline/perplexity.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── PPL computation ────────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 4,
    device: str = "cpu",
) -> list[float]:
    """
    Compute per-document perplexity under a causal language model.

    Uses a sliding window approach for documents longer than max_length tokens.
    This matches the standard evaluation protocol for autoregressive models.

    Args:
        model: HuggingFace AutoModelForCausalLM (must be in eval mode)
        tokenizer: corresponding tokenizer
        texts: list of document strings
        max_length: maximum context window
        stride: sliding window stride (overlap for long documents)
        device: 'cpu' or 'cuda'

    Returns:
        ppls: list of perplexity values, one per document
    """
    import torch

    model.eval()
    ppls = []

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 100 == 0 and len(texts) > 100:
                print(f"  PPL: {i}/{len(texts)} documents...")

            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
            )
            input_ids = encodings.input_ids.to(device)
            seq_len = input_ids.size(1)

            if seq_len == 0:
                ppls.append(float("nan"))
                continue

            nlls = []
            prev_end = 0

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                target_len = end - max(begin, prev_end)

                input_chunk = input_ids[:, begin:end]
                target_chunk = input_chunk.clone()

                # Mask out tokens already seen in previous windows
                if prev_end > begin:
                    target_chunk[:, : prev_end - begin] = -100

                output = model(input_chunk, labels=target_chunk)
                nlls.append(output.loss.item() * target_len)
                prev_end = end

                if end == seq_len:
                    break

            total_tokens = seq_len
            avg_nll = sum(nlls) / total_tokens if total_tokens > 0 else float("nan")
            ppl = float(np.exp(avg_nll))
            ppls.append(ppl)

    return ppls


def load_g0_model(model_id: str = "distilgpt2", device: str = "cpu"):
    """
    Load the pretrained G0 DistilGPT-2 model and tokenizer.

    This is the reference model — it must be in eval mode and never fine-tuned.
    If loading from a local checkpoint (Kaggle Dataset), pass the checkpoint path.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. pip install transformers==4.38.2")

    import torch

    print(f"Loading G0 model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    # Freeze — this model is read-only
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, tokenizer


# ── Baseline statistics ────────────────────────────────────────────────────────

def compute_ppl_baseline_stats(ppls: list[float]) -> dict:
    """
    Summarize the per-document PPL distribution.

    Expected range for DistilGPT-2 on Wikipedia: 15–30.
    Significantly higher suggests tokenization issues or domain mismatch.
    """
    ppls_clean = [p for p in ppls if not np.isnan(p) and np.isfinite(p)]

    if not ppls_clean:
        return {"error": "No valid PPL values computed"}

    ppls_arr = np.array(ppls_clean)

    # Histogram for distribution visualization (Phase 3 dashboard)
    hist, bin_edges = np.histogram(ppls_arr, bins=50)

    stats = {
        "n_documents": len(ppls),
        "n_valid": len(ppls_clean),
        "n_invalid": len(ppls) - len(ppls_clean),
        "mean_ppl": round(float(ppls_arr.mean()), 4),
        "std_ppl": round(float(ppls_arr.std()), 4),
        "median_ppl": round(float(np.median(ppls_arr)), 4),
        "pct5_ppl": round(float(np.percentile(ppls_arr, 5)), 4),
        "pct25_ppl": round(float(np.percentile(ppls_arr, 25)), 4),
        "pct75_ppl": round(float(np.percentile(ppls_arr, 75)), 4),
        "pct95_ppl": round(float(np.percentile(ppls_arr, 95)), 4),
        "min_ppl": round(float(ppls_arr.min()), 4),
        "max_ppl": round(float(ppls_arr.max()), 4),
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": [round(e, 4) for e in bin_edges.tolist()],
        },
        "sanity_check": {
            "expected_range": "15–30 for DistilGPT-2 on Wikipedia",
            "mean_in_range": 15.0 <= float(ppls_arr.mean()) <= 30.0,
            "diagnostic": (
                "pass" if 15.0 <= float(ppls_arr.mean()) <= 30.0
                else "warn_outside_expected_range"
            ),
        },
    }
    return stats


# ── Main computation ───────────────────────────────────────────────────────────

def compute_perplexity_baseline(
    documents: list[str],
    output_dir: Path,
    model_id: str = "distilgpt2",
    corpus_hash: str = "unknown",
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Compute and serialize the PPL baseline distribution under G0.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_g0_model(model_id, device)

    print(f"Computing PPL on {len(documents):,} documents...")
    ppls = compute_perplexity(model, tokenizer, documents, device=device)

    stats = compute_ppl_baseline_stats(ppls)

    if "sanity_check" in stats:
        sc = stats["sanity_check"]
        status = "OK" if sc["mean_in_range"] else "WARN"
        print(f"  [{status}] Mean PPL: {stats['mean_ppl']:.2f} ({sc['diagnostic']})")
        print(f"  Std PPL: {stats['std_ppl']:.2f}")

    ppl_baseline = {
        "_corpus_hash": corpus_hash,
        "_computed_at": __import__("datetime").datetime.now().isoformat(),
        "_model_id": model_id,
        **stats,
    }

    ppl_file = output_dir / "ppl_baseline.json"
    with open(ppl_file, "w") as f:
        json.dump(ppl_baseline, f, indent=2)
    print(f"  Written: {ppl_file}")

    return ppl_baseline


# ── Self-test ──────────────────────────────────────────────────────────────────

def run_self_test() -> bool:
    """
    Test PPL statistics computation without a real model.
    """
    print("\n── Perplexity Baseline Self-Test ─────────────────────────────────")
    all_passed = True

    # Synthetic PPL values matching expected Wikipedia/DistilGPT-2 range
    rng = np.random.default_rng(42)
    synthetic_ppls = rng.lognormal(mean=np.log(22), sigma=0.4, size=500).tolist()

    stats = compute_ppl_baseline_stats(synthetic_ppls)

    passed = 15 <= stats["mean_ppl"] <= 35
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Mean PPL in range: {stats['mean_ppl']:.2f}")

    passed = stats["std_ppl"] > 0
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Std PPL > 0: {stats['std_ppl']:.2f}")

    passed = stats["n_valid"] == 500
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] All documents valid: {stats['n_valid']}/500")

    passed = len(stats["histogram"]["counts"]) == 50
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] Histogram has 50 bins")

    # Test NaN handling
    ppls_with_nan = synthetic_ppls[:100] + [float("nan"), float("inf")]
    stats_nan = compute_ppl_baseline_stats(ppls_with_nan)
    passed = stats_nan["n_invalid"] == 2
    all_passed &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] NaN/inf handling: "
          f"{stats_nan['n_invalid']} invalid filtered")

    # Test serialization round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        result = {
            "_corpus_hash": "test",
            "_computed_at": "2026-01-01",
            "_model_id": "distilgpt2",
            **stats,
        }
        ppl_file = out / "ppl_baseline.json"
        with open(ppl_file, "w") as f:
            json.dump(result, f, indent=2)
        with open(ppl_file) as f:
            loaded = json.load(f)
        passed = abs(loaded["mean_ppl"] - stats["mean_ppl"]) < 1e-3
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] JSON round-trip: "
              f"mean_ppl={loaded['mean_ppl']:.4f}")

    print()
    print("  ✓ PERPLEXITY SELF-TEST PASSED" if all_passed
          else "  ✗ PERPLEXITY SELF-TEST FAILED")
    return all_passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO perplexity baseline computation")
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path(__file__).parent / "corpus")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "measurements")
    parser.add_argument("--model-id", type=str, default="distilgpt2")
    parser.add_argument("--device", type=str, default="cpu")
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

    compute_perplexity_baseline(
        documents=documents,
        output_dir=args.output_dir,
        model_id=args.model_id,
        corpus_hash=corpus_hash,
        device=args.device,
        seed=args.seed,
    )
