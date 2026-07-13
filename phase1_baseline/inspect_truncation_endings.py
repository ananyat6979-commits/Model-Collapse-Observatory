"""
EXP-010 (candidate) — Do at-cap documents end mid-sentence?

Manifest fact: 16,027 raw articles were TOO LONG and got truncated to
the 135-token cap (vs. 3,948 rejected for being too short). The
115-135 token bucket holds 3,646/5,000 (73%) of the final corpus
(EXP-008). This means most documents are very likely excerpts cut at
an arbitrary word boundary, not naturally-short complete articles.

The manual read in read_sample_docs.py only printed text[:400] — the
OPENING of each document. Nobody has looked at how documents actually
END, which is exactly where a truncation artifact would live. This
script does that, directly, with zero model inference.

Run from repo root. Instant, no GPU, no internet.
"""
import json
from pathlib import Path

CORPUS_PATH = Path("phase1_baseline/corpus/documents.jsonl")
N_EACH_GROUP = 10

docs = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        text = d.get("text", "")
        if text.strip():
            docs.append(text)

lengths = [(i, len(t.split())) for i, t in enumerate(docs)]

at_cap = [i for i, n in lengths if n >= 130][:N_EACH_GROUP]
under_cap = [i for i, n in lengths if 50 <= n <= 70][:N_EACH_GROUP]

def show_ending(label, indices):
    print(f"\n{'#'*70}\n{label}\n{'#'*70}")
    for idx in indices:
        text = docs[idx]
        n = len(text.split())
        tail = text[-200:]
        print(f"\n--- Doc #{idx} ({n} words) — last 200 chars ---")
        print(f"...{tail}")

show_ending(f"AT-CAP documents (>=130 tokens, presumed truncated) — n={len(at_cap)}", at_cap)
show_ending(f"UNDER-CAP documents (50-70 tokens, presumed complete) — n={len(under_cap)}", under_cap)

print(f"\n\n{'='*70}")
print("Read both groups above. Look specifically for:")
print("  - AT-CAP group: does text end mid-word, mid-sentence, or with no")
print("    terminal punctuation? That would confirm truncation artifacts.")
print("  - UNDER-CAP group: does text end naturally (period, 'References'")
print("    section, category tags)? That would confirm these are complete.")
print(f"{'='*70}")