"""
EXP-013: Exact truncation-tail contribution to whole-document PPL.

EXP-012 established, exactly, that removing the first 16 tokens of every
document reduces mean PPL from 44.4445 to 39.4612, a reduction of 4.98
points, 22.7 percent of the gap to the reference midpoint (22.5). That
closes the cold-start question with a real number. It does not close
the truncation-tail question. EXP-011 established the tail effect is
statistically real (p=7.2e-10) but never converted that into a
document-level PPL contribution. This script does that conversion,
exactly, the same way EXP-012 did for cold-start.

Method: for every document, compute per-token loss once (single
untruncated forward pass, matching production exactly). Then compute
three document-level means across the full 5000-document corpus:

  1. whole_document_mean_ppl        - reproduction check against 44.4445
  2. excl_coldstart_mean_ppl        - first 16 tokens removed (EXP-012
                                        reproduction check, must match
                                        39.4612 or something has drifted)
  3. excl_coldstart_and_tail_mean_ppl: first 16 tokens AND, for
     documents at or near the 135-token cap (>=130 whitespace tokens,
     the same threshold EXP-011 used), the last 5 tokens, removed.

The difference between (2) and (3) is the exact, non-estimated
truncation-tail contribution to whole-document PPL, isolated from
cold-start. This is not a re-estimate of the tail effect's
significance, that is already established. This is its magnitude.

Run from repo root. CPU only. Same cost as EXP-011/EXP-012, one
untruncated forward pass per document across the full corpus.

Produces: phase1_baseline/measurements/exp013_tail_contribution.json
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
COLDSTART_EXCLUDE = 16
TAIL_EXCLUDE = 5
AT_CAP_MIN_TOKENS = 130  # matches EXP-011's threshold, do not change without
                          # re-justifying against manifest.json truncation stats

REFERENCE_WHOLE_DOC_PPL = 44.4445
REFERENCE_EXCL_COLDSTART_PPL = 39.4612  # EXP-012, must reproduce


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

    whole_doc_losses = []
    excl_coldstart_losses = []
    excl_coldstart_and_tail_losses = []
    n_at_cap = 0
    n_under_cap = 0

    print(f"\nComputing exact per-document PPL under three exclusion "
          f"conditions for {len(documents)} documents...")
    t0 = time.time()
    for i, text in enumerate(documents):
        if i % 500 == 0:
            print(f"  {i}/{len(documents)}...")
        losses = per_token_losses(model, tokenizer, text)
        if losses is None or len(losses) == 0:
            continue

        n_words = len(text.split())
        is_at_cap = n_words >= AT_CAP_MIN_TOKENS
        if is_at_cap:
            n_at_cap += 1
        else:
            n_under_cap += 1

        # Whole document, unmodified.
        whole_doc_losses.append(float(np.mean(losses)))

        # Cold-start removed. If a document is too short to have tokens
        # past position 16, it cannot contribute to this condition,
        # matching EXP-012's exclusion rule exactly.
        if len(losses) > COLDSTART_EXCLUDE:
            excl_coldstart_losses.append(float(np.mean(losses[COLDSTART_EXCLUDE:])))

        # Cold-start removed AND, for at-cap documents only, the last
        # TAIL_EXCLUDE tokens also removed. Under-cap documents are
        # presumed complete (EXP-011 manual inspection: 0/10 showed
        # truncation artifacts) and are not tail-trimmed here, since
        # trimming a complete document's real ending is not the effect
        # being isolated.
        if is_at_cap:
            end = len(losses) - TAIL_EXCLUDE
            if end > COLDSTART_EXCLUDE:
                excl_coldstart_and_tail_losses.append(
                    float(np.mean(losses[COLDSTART_EXCLUDE:end]))
                )
        else:
            if len(losses) > COLDSTART_EXCLUDE:
                excl_coldstart_and_tail_losses.append(
                    float(np.mean(losses[COLDSTART_EXCLUDE:]))
                )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")
    print(f"  at_cap documents: {n_at_cap}   under_cap-or-mid documents: {n_under_cap}")

    whole_doc_mean_ppl = float(np.exp(np.mean(whole_doc_losses)))
    excl_coldstart_mean_ppl = float(np.exp(np.mean(excl_coldstart_losses)))
    excl_both_mean_ppl = float(np.exp(np.mean(excl_coldstart_and_tail_losses)))

    coldstart_reduction = whole_doc_mean_ppl - excl_coldstart_mean_ppl
    tail_reduction = excl_coldstart_mean_ppl - excl_both_mean_ppl
    total_reduction = whole_doc_mean_ppl - excl_both_mean_ppl

    reference_midpoint = 22.5
    total_gap = whole_doc_mean_ppl - reference_midpoint
    remaining_gap = excl_both_mean_ppl - reference_midpoint

    print("\n" + "=" * 65)
    print("EXACT TAIL CONTRIBUTION RESULTS")
    print("=" * 65)
    print(f"  Whole-document mean PPL (reproduction check vs {REFERENCE_WHOLE_DOC_PPL}): "
          f"{whole_doc_mean_ppl:.4f}")
    coldstart_repro_pass = abs(excl_coldstart_mean_ppl - REFERENCE_EXCL_COLDSTART_PPL) < 0.05
    print(f"  Excl-coldstart mean PPL (reproduction check vs {REFERENCE_EXCL_COLDSTART_PPL}): "
          f"{excl_coldstart_mean_ppl:.4f}  [{'PASS' if coldstart_repro_pass else 'FAIL, INVESTIGATE'}]")
    print(f"  Excl-coldstart-and-tail mean PPL: {excl_both_mean_ppl:.4f}")
    print()
    print(f"  Cold-start reduction (EXP-012 quantity): {coldstart_reduction:.4f} PPL points")
    print(f"  Additional tail-exclusion reduction (NEW, this script): {tail_reduction:.4f} PPL points")
    print(f"  Total reduction from both mechanisms: {total_reduction:.4f} PPL points")
    print()
    print(f"  Total gap to reference midpoint ({reference_midpoint}): {total_gap:.4f} PPL points")
    print(f"  Gap explained by cold-start alone: "
          f"{100 * coldstart_reduction / total_gap:.1f} percent")
    print(f"  Gap explained by cold-start plus tail-trimming: "
          f"{100 * total_reduction / total_gap:.1f} percent")
    print(f"  Remaining unexplained gap after both mechanisms: "
          f"{remaining_gap:.4f} PPL points above reference midpoint")
    print("=" * 65)
    print()
    print("This remaining gap is UNEXPLAINED. Do not attribute it to genre or")
    print("content-difficulty differences or any other mechanism without a")
    print("dedicated test. State it as unexplained in THREATS_TO_VALIDITY.md.")

    out_path = Path(__file__).parent / "measurements" / "exp013_tail_contribution.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment_id": "EXP-013",
            "purpose": "Exact, non-estimated quantification of the truncation-tail "
                       "contribution to whole-document PPL, completing the "
                       "decomposition EXP-012 began for cold-start alone.",
            "n_documents": len(documents),
            "n_at_cap": n_at_cap,
            "n_under_cap_or_mid": n_under_cap,
            "coldstart_exclude_tokens": COLDSTART_EXCLUDE,
            "tail_exclude_tokens": TAIL_EXCLUDE,
            "at_cap_min_tokens_threshold": AT_CAP_MIN_TOKENS,
            "whole_document_mean_ppl": round(whole_doc_mean_ppl, 4),
            "excl_coldstart_mean_ppl": round(excl_coldstart_mean_ppl, 4),
            "excl_coldstart_and_tail_mean_ppl": round(excl_both_mean_ppl, 4),
            "coldstart_reproduction_check_passed": coldstart_repro_pass,
            "coldstart_reduction_ppl_points": round(coldstart_reduction, 4),
            "additional_tail_reduction_ppl_points": round(tail_reduction, 4),
            "total_reduction_ppl_points": round(total_reduction, 4),
            "reference_midpoint": reference_midpoint,
            "total_gap_ppl_points": round(total_gap, 4),
            "pct_gap_explained_coldstart_only": round(100 * coldstart_reduction / total_gap, 2),
            "pct_gap_explained_coldstart_and_tail": round(100 * total_reduction / total_gap, 2),
            "remaining_unexplained_gap_ppl_points": round(remaining_gap, 4),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()