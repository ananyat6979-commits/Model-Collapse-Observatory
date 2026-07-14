"""
EXP-012: Exact per-document PPL reconstruction from position-effect data.

Resolves the estimation gap in EXP-011's conclusion. EXP-011's write-up
used an informal back-of-envelope calculation ("15 tokens at loss 4.9955,
remaining ~165 tokens at plateau ~3.55") to estimate that the cold-start
effect alone would produce document-level PPL around 39-40. This script
computes the ACTUAL exact value from the real per-document per-token
losses already gathered in EXP-011, no estimation, no eyeballing.

This also directly tests whether the "residual ~5 PPL points" claim in
the EXP-011 write-up is real or an artifact of the informal estimate.

Requires: EXP-011 must have been run first (this script recomputes from
scratch using the same per-token loss method, not by reading
ppl_position_effect.json, since that file only stores bucketed
aggregates, not per-document values).

Run from repo root. Same cost as EXP-011 (~5000 forward passes, CPU).
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
REFERENCE_MEAN_PPL = 44.4445  # from ppl_baseline.json, the number we are trying to explain


def per_token_losses(model, tokenizer, text, device="cpu"):
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
    print("Loading corpus (hash-verified)...")
    documents = load_corpus(CORPUS_DIR)
    print(f"  {len(documents)} documents loaded")

    print(f"\nLoading {MODEL_ID}...")
    model, tokenizer = load_g0_model(MODEL_ID, device="cpu")

    per_doc_mean_ppl = []
    per_doc_ppl_excluding_first16 = []
    per_doc_n_tokens = []

    print(f"\nComputing exact per-document PPL, both whole-document and")
    print(f"excluding-first-16-tokens variants, for {len(documents)} documents...")
    t0 = time.time()
    for i, text in enumerate(documents):
        if i % 500 == 0:
            print(f"  {i}/{len(documents)}...")
        losses = per_token_losses(model, tokenizer, text)
        if losses is None or len(losses) == 0:
            continue

        n_tok = len(losses)
        per_doc_n_tokens.append(n_tok)

        # Whole-document mean PPL (this should reproduce 44.4445 exactly)
        whole_doc_ppl = float(np.exp(np.mean(losses)))
        per_doc_mean_ppl.append(whole_doc_ppl)

        # PPL excluding the first 16 tokens (positions 2-17), i.e. the
        # cold-start region only, testing whether removing it alone
        # explains the gap, computed EXACTLY not estimated
        if n_tok > 16:
            ppl_excl = float(np.exp(np.mean(losses[16:])))
            per_doc_ppl_excluding_first16.append(ppl_excl)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    whole_doc_mean = float(np.mean(per_doc_mean_ppl))
    excl_first16_mean = float(np.mean(per_doc_ppl_excluding_first16))

    print("\n" + "=" * 65)
    print("EXACT RECONSTRUCTION RESULTS")
    print("=" * 65)
    print(f"  Whole-document mean PPL (should match 44.4445 exactly): "
          f"{whole_doc_mean:.4f}")
    print(f"  Reproduction check: "
          f"{'PASS' if abs(whole_doc_mean - REFERENCE_MEAN_PPL) < 0.01 else 'FAIL - investigate'}")
    print()
    print(f"  Mean PPL EXCLUDING first 16 tokens (cold-start removed): "
          f"{excl_first16_mean:.4f}")
    print(f"  This is the EXACT value, not an estimate.")
    print()

    gap_original = whole_doc_mean - 15  # rough midpoint reference, adjust as needed
    gap_after_removal = excl_first16_mean - 15
    pct_explained_by_coldstart = (
        (whole_doc_mean - excl_first16_mean) / (whole_doc_mean - 22.5) * 100
        if whole_doc_mean != 22.5 else None
    )

    print(f"  Whole-document PPL: {whole_doc_mean:.2f}")
    print(f"  PPL with cold-start removed: {excl_first16_mean:.2f}")
    print(f"  Reduction from removing cold-start alone: "
          f"{whole_doc_mean - excl_first16_mean:.2f} PPL points")
    print(f"  Midpoint of documented reference range (15-30): 22.5")
    print(f"  Remaining gap after cold-start removal: "
          f"{excl_first16_mean - 22.5:.2f} PPL points above reference midpoint")

    out_path = Path(__file__).parent / "measurements" / "exp012_exact_reconstruction.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment_id": "EXP-012",
            "purpose": "Exact (not estimated) computation of document-level PPL "
                       "with cold-start region removed, to replace the informal "
                       "back-of-envelope estimate in EXP-011's write-up.",
            "n_documents": len(per_doc_mean_ppl),
            "whole_document_mean_ppl": round(whole_doc_mean, 4),
            "reference_baseline_mean_ppl": REFERENCE_MEAN_PPL,
            "reproduction_check_passed": bool(abs(whole_doc_mean - REFERENCE_MEAN_PPL) < 0.01),
            "mean_ppl_excluding_first_16_tokens": round(excl_first16_mean, 4),
            "reduction_from_coldstart_removal": round(whole_doc_mean - excl_first16_mean, 4),
            "reference_range_midpoint": 22.5,
            "remaining_gap_after_coldstart_removal": round(excl_first16_mean - 22.5, 4),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nUse the EXACT values above, not the EXP-011 write-up's estimated")
    print("~39-40 figure, when writing THREATS_TO_VALIDITY.md.")


if __name__ == "__main__":
    main()