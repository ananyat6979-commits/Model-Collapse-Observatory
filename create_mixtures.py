"""
Creates five synthetic/human mixtures at known contamination fractions:
  0%, 25%, 50%, 75%, 100% synthetic

Uses G3 outputs (most collapsed, R=0.5) as the synthetic source.
Human docs come from a held-out 20% of the Phase 1 Wikipedia corpus
(different random seed from the training split to avoid overlap).

Hypothesis (pre-experiment):
  composite_index should increase monotonically with synthetic fraction.
  Expected: 0pct ~0.02, 25pct ~0.15, 50pct ~0.35, 75pct ~0.55, 100pct ~0.65

If confirmed, this is the first empirical calibration of a collapse
measurement framework against known ground truth fractions.

Usage:
    python create_mixtures.py

Pre-condition:
    Download G3_outputs/synthetic.jsonl from Kaggle dataset
    ananyatiwari0212/mco-phase2-artifacts and place at:
    phase2_simulation/G3_outputs/synthetic.jsonl
"""

import json
import random
import sys
from pathlib import Path

SEED     = 42
N_TOTAL  = 1000
CORPUS   = Path("phase1_baseline/corpus/documents.jsonl")
G3_OUT   = Path("phase2_simulation/G3_outputs/synthetic.jsonl")
OUT_DIR  = Path("phase5_index/calibration")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_text(doc):
    return doc.get("text", "") if isinstance(doc, dict) else str(doc)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("Loading human corpus...")
    human_docs = load_jsonl(CORPUS)
    print(f"  {len(human_docs)} human documents loaded")

    if not G3_OUT.exists():
        print(f"\nERROR: {G3_OUT} not found.")
        print("Download from Kaggle: ananyatiwari0212/mco-phase2-artifacts")
        print(f"Place at: {G3_OUT}")
        sys.exit(1)

    print("Loading G3 synthetic outputs...")
    synth_docs = load_jsonl(G3_OUT)
    print(f"  {len(synth_docs)} synthetic documents (G3, R=0.5, DistilGPT-2)")

    # Holdout split: use different seed from training to avoid overlap
    all_idx = list(range(len(human_docs)))
    rng_holdout = random.Random(SEED + 999)
    holdout_idx = set(rng_holdout.sample(all_idx, len(all_idx) // 5))
    holdout = [human_docs[i] for i in sorted(holdout_idx)]
    print(f"  Human holdout (20%): {len(holdout)} docs")

    assert len(holdout)  >= N_TOTAL, f"Need {N_TOTAL} holdout docs, have {len(holdout)}"
    assert len(synth_docs) >= N_TOTAL, f"Need {N_TOTAL} synth docs, have {len(synth_docs)}"

    fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = []

    for frac in fractions:
        n_synth = int(N_TOTAL * frac)
        n_human = N_TOTAL - n_synth

        sample_h = rng.sample(holdout,     n_human)
        sample_s = rng.sample(synth_docs,  n_synth)

        mixture = (
            [{"text": get_text(d), "source": "human",        "true_synthetic_fraction": frac} for d in sample_h] +
            [{"text": get_text(d), "source": "synthetic_g3", "true_synthetic_fraction": frac} for d in sample_s]
        )
        rng.shuffle(mixture)

        frac_pct = int(frac * 100)
        out_path = OUT_DIR / f"mixture_{frac_pct:03d}pct_synthetic.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in mixture:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        results.append({"fraction": frac, "n_human": n_human, "n_synth": n_synth, "path": str(out_path)})
        print(f"  {frac_pct:3d}% synthetic: {n_human} human + {n_synth} synthetic -> {out_path.name}")

    # Write manifest
    manifest_path = OUT_DIR / "calibration_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_total_per_mixture": N_TOTAL,
            "synthetic_source": "G3_outputs (DistilGPT-2, R=0.5, k=3)",
            "human_source": "Wikipedia holdout 20% (seed=42+999, no overlap with training)",
            "hypothesis": "composite_index increases monotonically with synthetic fraction",
            "expected": {"0pct": 0.02, "25pct": 0.15, "50pct": 0.35, "75pct": 0.55, "100pct": 0.65},
            "mixtures": results,
        }, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"\nNext: python score_mixtures.py")
    print("This will score each mixture and produce the calibration curve.")


if __name__ == "__main__":
    main()