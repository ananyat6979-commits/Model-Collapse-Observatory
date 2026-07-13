"""
EXP-011 — Per-token-position loss curve: does PPL track context length,
not document length?

Motivation (see references/experiment_log.md EXP-008/010 write-ups):
  - EXP-008: whole-document mean PPL does NOT correlate meaningfully with
    document length within this corpus (rho=-0.036). The 115-135 token
    bucket (likely truncated, per manifest.json: 16,027 raw articles were
    cut to fit the cap) is NOT the most elevated bucket; the 0-74 token
    bucket is (50.24 vs 44.08).
  - Manual ending inspection (inspect_truncation_endings.py): 6/10 at-cap
    documents (>=130 tokens) show unambiguous mid-sentence/mid-word
    truncation artifacts; 0/10 under-cap documents (50-70 tokens) do.
    Truncation is real and common, but does NOT explain the whole-
    document PPL pattern above — a direct tension worth resolving.

Reconciling hypothesis: autoregressive LM loss is known to be higher on
early-position tokens (less preceding context). Since EVERY document in
this corpus is capped at <=135 tokens (vs. likely thousands of tokens in
whatever full-article corpus the "15-30" reference range came from), a
much larger PROPORTION of every document's tokens sit in the
high-loss, low-context "early position" region. This would explain:
  (a) why the corpus sits uniformly ~40-44 regardless of INTERNAL length
      variation (all documents are short in absolute terms),
  (b) why the shortest bucket is MOST elevated (highest proportion of
      early-position tokens),
  (c) without contradicting the truncation-ending finding, which is a
      separate, likely small, tail-concentrated effect.

This script tests (a)/(b) directly by computing mean loss as a function
of ABSOLUTE TOKEN POSITION within each document, aggregated across the
whole corpus, plus a secondary check comparing last-5-token loss between
at-cap (likely-truncated) and under-cap (likely-complete) documents to
isolate the truncation-specific tail effect from the general
context-length effect.

No sliding window needed here (confirmed in EXP-008/perplexity.py that
it never triggers for documents this short); a single untruncated
forward pass per document is used, matching production behavior exactly.

Run from repo root. CPU only. ~15-20 min for 5000 docs (same cost class
as EXP-008, one forward pass per document).

Produces: phase1_baseline/measurements/ppl_position_effect.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase1_baseline.perplexity import load_g0_model  # noqa: E402
from phase1_baseline.corpus.ingest import load_corpus  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
MODEL_ID = "distilgpt2"
POSITION_BUCKET_SIZE = 15  # positions 1-15, 16-30, ... up to 135
TAIL_K = 5  # last-K-token comparison for truncation-specific check
AT_CAP_MIN_TOKENS = 130
UNDER_CAP_RANGE = (50, 70)


def per_token_losses(model, tokenizer, text, device="cpu"):
    """
    Single untruncated forward pass. Returns a 1D array of per-token
    losses, where losses[i] is the loss for predicting token i+1 given
    tokens 0..i (standard causal-LM shift). losses[i] corresponds to
    ABSOLUTE POSITION i+2 in the document (1-indexed): the first token
    has no prediction target (nothing precedes it), so losses[0] is the
    loss at position 2.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=False).to(device)
    input_ids = enc.input_ids
    if input_ids.size(1) < 2:
        return None
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        )
    return losses.cpu().numpy()


def main():
    print("Loading corpus (hash-verified via load_corpus)...")
    documents = load_corpus(CORPUS_DIR)
    print(f"  {len(documents)} documents loaded")

    print(f"\nLoading {MODEL_ID}...")
    model, tokenizer = load_g0_model(MODEL_ID, device="cpu")

    # position -> list of losses across all documents
    position_losses: dict[int, list] = {}
    at_cap_tail_losses = []
    under_cap_tail_losses = []
    doc_lengths = []

    print(f"\nComputing per-token losses for {len(documents)} documents "
          f"(single untruncated forward pass each)...")
    t0 = time.time()
    for i, text in enumerate(documents):
        if i % 500 == 0:
            print(f"  {i}/{len(documents)}...")
        losses = per_token_losses(model, tokenizer, text)
        if losses is None or len(losses) == 0:
            continue
        n_words = len(text.split())
        doc_lengths.append(n_words)

        for j, loss_val in enumerate(losses):
            pos = j + 2  # absolute 1-indexed position of the predicted token
            position_losses.setdefault(pos, []).append(float(loss_val))

        if n_words >= AT_CAP_MIN_TOKENS and len(losses) >= TAIL_K:
            at_cap_tail_losses.extend(losses[-TAIL_K:].tolist())
        elif UNDER_CAP_RANGE[0] <= n_words <= UNDER_CAP_RANGE[1] and len(losses) >= TAIL_K:
            under_cap_tail_losses.extend(losses[-TAIL_K:].tolist())

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── Bucket by position, report mean loss and implied perplexity ────
    max_pos = max(position_losses.keys())
    bucket_edges = list(range(2, max_pos + POSITION_BUCKET_SIZE + 1, POSITION_BUCKET_SIZE))
    position_bucket_stats = {}
    print(f"\n{'Position bucket':>18} | {'n_tokens':>10} | {'mean_loss':>10} | {'implied_ppl':>12}")
    print("-" * 58)
    for lo in bucket_edges:
        hi = lo + POSITION_BUCKET_SIZE
        bucket_losses = []
        for pos in range(lo, hi):
            bucket_losses.extend(position_losses.get(pos, []))
        if not bucket_losses:
            continue
        mean_loss = float(np.mean(bucket_losses))
        implied_ppl = float(np.exp(mean_loss))
        label = f"{lo}-{hi - 1}"
        position_bucket_stats[label] = {
            "n_tokens": len(bucket_losses),
            "mean_loss": round(mean_loss, 4),
            "implied_ppl": round(implied_ppl, 4),
        }
        print(f"{label:>18} | {len(bucket_losses):>10} | {mean_loss:>10.4f} | {implied_ppl:>12.2f}")

    bucket_ppls = [v["implied_ppl"] for v in position_bucket_stats.values()]
    monotonic_decreasing = all(
        bucket_ppls[i] >= bucket_ppls[i + 1] for i in range(len(bucket_ppls) - 1)
    )

    # ── Tail-effect comparison (truncation-specific) ────────────────────
    tail_comparison = {}
    if at_cap_tail_losses and under_cap_tail_losses:
        at_cap_mean = float(np.mean(at_cap_tail_losses))
        under_cap_mean = float(np.mean(under_cap_tail_losses))
        from scipy import stats as scipy_stats
        u_stat, p_val = scipy_stats.mannwhitneyu(
            at_cap_tail_losses, under_cap_tail_losses, alternative="greater"
        )
        tail_comparison = {
            "at_cap_n_tokens": len(at_cap_tail_losses),
            "at_cap_mean_loss": round(at_cap_mean, 4),
            "at_cap_implied_ppl": round(float(np.exp(at_cap_mean)), 4),
            "under_cap_n_tokens": len(under_cap_tail_losses),
            "under_cap_mean_loss": round(under_cap_mean, 4),
            "under_cap_implied_ppl": round(float(np.exp(under_cap_mean)), 4),
            "mann_whitney_u": float(u_stat),
            "p_value_at_cap_greater": float(p_val),
        }
        print(f"\nTail-{TAIL_K}-token comparison (truncation-specific check):")
        print(f"  At-cap (>={AT_CAP_MIN_TOKENS} tok, likely truncated):   "
              f"mean_loss={at_cap_mean:.4f}  implied_ppl={np.exp(at_cap_mean):.2f}  "
              f"n={len(at_cap_tail_losses)}")
        print(f"  Under-cap ({UNDER_CAP_RANGE[0]}-{UNDER_CAP_RANGE[1]} tok, likely complete): "
              f"mean_loss={under_cap_mean:.4f}  implied_ppl={np.exp(under_cap_mean):.2f}  "
              f"n={len(under_cap_tail_losses)}")
        print(f"  Mann-Whitney (at-cap tail > under-cap tail): p={p_val:.6f}")

    # ── Verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if monotonic_decreasing:
        print("POSITION HYPOTHESIS SUPPORTED: implied PPL decreases monotonically")
        print("with token position (more preceding context = lower loss). This is")
        print("consistent with the corpus's short absolute length (<=135 tokens)")
        print("explaining the elevated 44.4 baseline via low-average-context-per-")
        print("token, distinct from the internal length-bucket comparison in EXP-008.")
    else:
        print("POSITION HYPOTHESIS NOT CLEANLY SUPPORTED: implied PPL does not")
        print("decrease monotonically with position. Re-examine bucket table above.")
    print("=" * 65)

    out_path = Path(__file__).parent / "measurements" / "ppl_position_effect.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment_id": "EXP-011",
            "purpose": "Test whether low context-per-token (a consequence of this "
                       "corpus's <=135 token cap) explains the 44.4 vs 15-30 gap, "
                       "reconciling EXP-008's flat length-PPL correlation with the "
                       "visually-confirmed truncation artifacts in at-cap documents.",
            "n_documents": len(doc_lengths),
            "position_bucket_size": POSITION_BUCKET_SIZE,
            "position_buckets": position_bucket_stats,
            "position_ppl_monotonic_decreasing": monotonic_decreasing,
            "tail_truncation_comparison": tail_comparison,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()