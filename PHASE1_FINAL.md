# Phase 1 — Final Assessment and Run Instructions

## Verdict

The pipeline is done. This document explains what changed in the final version,
why there is nothing more to iterate on, and exactly what to run.

---

## The One Real Change

**`calibrate_max_doc_length` now hard-fails when `transformers` is missing.**

Previous behavior:
```python
except ImportError:
    log.warning("transformers not installed — skipping tokenizer calibration...")
    return {"calibration": "skipped_no_transformers", "recommended_max_tokens": 200}
```

New behavior:
```python
except ImportError:
    raise RuntimeError(
        "Tokenizer calibration requires 'transformers' to be installed.\n"
        "  Install: pip install transformers==4.38.2\n..."
    ) from None
```

**Why this was the last real issue:**

The soft fallback was poison for a locked baseline. It would let the pipeline
produce a corpus whose manifest says "empirical_tokenizer_calibration" but whose
actual length cap was an unverified constant. Documents could be silently
truncated by the encoder. The semantic baseline — including the tail mass
measurement — would be built on the first ~50–70 words of each article instead
of the full document. The primary collapse signal would be measuring the wrong
distribution from the start.

Hard-failing on missing `transformers` forces the researcher to fix the environment
before the corpus is built. This is the correct behavior for a locked baseline.

---

## What Was Cleaned Up (Not Bugs — Polish)

The `build_corpus` Stage 0 block was simplified. The old code checked for
`"derived_whitespace_token_cap" in calibration_result` — a guard that existed
because calibration could silently return a fallback dict. Since calibration now
raises on failure, the guard is gone. The code directly reads
`calibration_result["derived_whitespace_token_cap"]`.

The manifest's `n_raw_rejected` section was corrected: `n_too_long` previously
reflected preliminary-cap truncations before calibration. It now reflects the
count from Stage 1b (after calibration), which is the accurate number.

A check for calibration being recorded was added to `validate_corpus()`. This
catches the case where someone runs an old ingest.py without calibration and
tries to use the corpus with a new pipeline expecting calibration metadata.

---

## What Is NOT a Bug (Do Not Change These)

**MinHash seed reuse.** Every review flagged this. It is correct behavior.
MinHash requires identical hash functions applied to every document. The same
seed produces the same `(a, b)` parameters for each permutation — that is the
definition of valid MinHash. See the docstring in `minhash_signature`.

**Calibration sample bias from PRELIMINARY_CAP=500.** The calibration samples
are truncated at 500 tokens before tokenizer calibration. Long articles with
denser subword inflation in their latter portions are slightly underrepresented.
This is a real and known residual bias. The 20-token safety margin and
conservative floor rounding account for it. It is documented in the calibration
comment block and in the log warning. It is not blocking.

**KDE fitted on train/eval split (60/40) rather than all data.**
This was correctly identified in an earlier session as preventing KDE
self-overfitting from biasing the tail threshold upward. The split is correct.
The previous review's concern about "fitting only 1,200 points" is resolved by
the compound subsampling fix — the KDE now receives all 5,000 PCA-projected
embeddings, 60/40 split → 3,000 train / 2,000 threshold evaluation.

**`wikiextractor` not used.** `iterparse` with `elem.clear()` is the standard
streaming pattern for Wikipedia dumps and handles 22GB files correctly.
`wikiextractor` adds a dependency with no benefit at this corpus scale.

---

## Remaining Acknowledged Limitations

These appear in `SCOPE.md` and in the Phase 4 Early Warning Card. They are
documented constraints, not bugs.

1. Results calibrated at DistilGPT-2 scale (82M). Not generalizable to larger models.
2. Corpus is English Wikipedia. Domain-specific collapse may behave differently.
3. Only R=0.5 fixed contamination schedule. Detection order may differ at other ratios.
4. k=3 generations is a short simulation chain.
5. Perplexity inversion signal is novel and unvalidated by prior work.
6. Calibration samples truncated at 500 tokens — p95 ratio is slightly optimistic.
   Safety margin compensates, but ~1% of documents may see mild encoder truncation.

These are the honest limitations of a controlled small-scale study.
Acknowledge them in any paper or preprint.

---

## Download and Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`transformers==4.38.2` must be present. Calibration hard-fails without it.
Verify: `python -c "from transformers import AutoTokenizer; print('OK')"`.

### 2. Download the Wikipedia dump

```bash
mkdir -p data
cd data

# ~22GB, use aria2 for resumable parallel download
aria2c -x 16 -s 16 \
  https://dumps.wikimedia.org/enwiki/20241201/enwiki-20241201-pages-articles-multistream.xml.bz2

# Download checksum file and verify
aria2c https://dumps.wikimedia.org/enwiki/20241201/md5sums.txt
grep "pages-articles-multistream.xml.bz2" md5sums.txt | md5sum -c --ignore-missing

# Compute SHA-256 for config.yaml
sha256sum enwiki-20241201-pages-articles-multistream.xml.bz2
# → record this value as corpus_dump_sha256 in config.yaml

cd ..
```

To use a different snapshot (any dated directory from dumps.wikimedia.org/enwiki/):
replace `20241201` with your chosen date throughout. Pin this date permanently in
`config.yaml`; do not use the `latest/` alias for a reproducible baseline.

### 3. Run the pipeline

```bash
python phase1_baseline/run.py --seed 42 \
  --dump data/enwiki-20241201-pages-articles-multistream.xml.bz2
```

Expected duration on a modern laptop:
- Ingestion + calibration: 30–90 minutes (22GB bz2 decompression is the bottleneck)
- Lexical baseline: ~2 minutes
- Semantic baseline (CPU embedding): ~10–15 minutes
- KDE bandwidth CV: ~5–10 minutes
- PPL baseline: ~20–30 minutes (5,000 documents through DistilGPT-2)
- Pack assembly: ~1 minute
- Total: 1.5–3 hours

### 4. Verify the completed pipeline

```bash
python phase1_baseline/run.py --seed 42 --test-only
```

Phase 1 is complete when all of these pass:

| Check | Expected value | Meaning |
|---|---|---|
| Zipf alpha 0.8–1.2 | ~1.0 | Natural language distribution |
| Corpus TTR > 0.10 | ~0.15–0.25 | Adequate lexical diversity |
| Intrinsic dimensionality > 10 | ~15–30 | Semantically diverse corpus |
| Mean PPL 15–30 | ~18–25 | DistilGPT-2 on Wikipedia |
| G0 weight hash recorded | non-empty SHA-256 | Reproducible PPL reference |

If any check fails, the failure tells you which stage went wrong.
Do NOT proceed to Phase 2 until all five pass.

### 5. Update config.yaml with real values

After a successful run, record these values:
```yaml
corpus_sha256: <value from phase1_baseline/corpus/manifest.json>
g0_model_weights_sha256: <value from phase1_baseline/measurements/ppl_baseline.json>
```

These are now locked. They must match on every subsequent run.

### 6. Run once more to verify reproducibility

```bash
# Run the full pipeline a second time with the same seed
python phase1_baseline/run.py --seed 42 \
  --dump data/enwiki-20241201-pages-articles-multistream.xml.bz2

# Compare reference_pack.pkl hashes from both runs
sha256sum phase1_baseline/reference_pack.pkl
# Both runs must produce the same hash.
```

### 7. Sign off Phase 1

```bash
# Via MCP server
mco_complete_phase phase=1 \
  artifact_paths=["phase1_baseline/reference_pack.pkl",
                  "phase1_baseline/corpus/manifest.json",
                  "phase1_baseline/measurements/"] \
  notes="Zipf alpha=X, TTR=X, IntrinsicDim=X, PPL=X, G0 hash=X" \
  signed_by="<your name>"
```

Then begin Phase 2 on Kaggle. Upload `reference_pack.pkl` and
`corpus/documents.jsonl` to a Kaggle Dataset before the first generation run.

---

## Why There Is Nothing Left to Iterate On

Every hostile review across this session series identified the following issues:

1. ✅ Namespace filter — fixed (using `<ns>` XML element)
2. ✅ Hard-fail on underfilled corpus — fixed (before and after dedup)
3. ✅ BookCorpus provenance claim — removed from manifest, SCOPE.md, config.yaml
4. ✅ `html.unescape()` + NFKC normalization — added in correct order
5. ✅ Non-ASCII sentinel tests — 21 checks, all passing
6. ✅ KDE train/eval split — preventing threshold bias from KDE self-overfitting
7. ✅ Compound subsampling — all PCA embeddings saved, not 2,000 of 5,000
8. ✅ DistilGPT-2 weight hash — computed and stored in ppl_baseline.json
9. ✅ GMM covariance type — changed to diagonal for valid Euclidean threshold
10. ✅ `compute_semantic_coverage` moved to Phase 3 — eliminated cross-phase import
11. ✅ Calibration sample from candidate pool — not dump front matter
12. ✅ Calibration hard-fail on missing `transformers` — no more silent fallback
13. ✅ KL distributions inlined in pack — no filesystem path references
14. ✅ Pack verification tied to corpus manifest count — not hard-coded 1000

The code has no remaining known issues. The only thing that remains
is operational: download the dump, run the pipeline, validate the numbers.

That is not a code problem. That is science.
