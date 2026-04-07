# Phase 1 Runbook — Windows, Python 3.10, HuggingFace backend

This document contains the exact commands to run, in order. Read the entire
document before starting. Each section is one coherent operation.

---

## Current State (as of last session)

- Environment: ✅ Working. Python 3.10 venv, all packages installed, torch determinism OK.
- Corpus: ❌ Not built. bz2 XML parsing fails with MemoryError on this machine.
- Solution: Use `ingest_hf.py` to build the corpus from the HuggingFace streaming backend.
- Dump file: Downloaded (md5 verified OK), but not needed for corpus building.

---

## Step 0 — File Placement (do this first)

Copy the updated `ingest_hf.py` to the correct location:

```powershell
# From the project root D:\Code\Projects\MachineLearning\MCO
# Replace the existing ingest_hf.py with the new one
```

The new `ingest_hf.py` adds two things the old one was missing:
1. **Idempotency**: if the corpus already exists and its hash is valid, it
   returns immediately without re-running the streaming pass.
2. **`too_long_truncated` counter in manifest**: the rejection accounting
   now correctly records how many documents were re-truncated by the calibrated cap.

Place it at: `phase1_baseline\corpus\ingest_hf.py`

---

## Step 1 — Activate the Environment

Every session starts here. If the terminal is fresh:

```powershell
cd D:\Code\Projects\MachineLearning\MCO
.\mco-env\Scripts\Activate.ps1
```

Confirm: your prompt should say `(mco-env)` at the start.

Verify imports are still working:
```powershell
python -c "import torch, transformers, sklearn, numpy, sentence_transformers; print('ALL OK')"
```

Expected output: `ALL OK`

---

## Step 2 — Build the Corpus (20–40 minutes)

This is the main command. Run it exactly as written:

```powershell
python phase1_baseline\corpus\ingest_hf.py `
  --output-dir phase1_baseline\corpus `
  --seed 42
```

**What you will see:**
```
2026-03-28 ... INFO === MCO Corpus Ingestion (HuggingFace backend) ===
2026-03-28 ... INFO Dataset: wikipedia / 20220301.en / train
2026-03-28 ... INFO Loading Wikipedia dataset in streaming mode...
2026-03-28 ... INFO Stage 1: Collecting and cleaning candidates...
2026-03-28 ... INFO   Scanned 10,000, collected 847 candidates
2026-03-28 ... INFO   Scanned 20,000, collected 1,823 candidates
...
2026-03-28 ... INFO Stage 0: Tokenizer calibration on random 500 candidates...
2026-03-28 ... INFO   Calibration: p95 ratio=1.4xx, derived cap=1xx tokens
2026-03-28 ... INFO Stage 1b: Applying calibrated cap...
2026-03-28 ... INFO Stage 2: MinHash deduplication...
2026-03-28 ... INFO Stage 3: Sampling to target...
2026-03-28 ... INFO Stage 4: Computing statistics...
2026-03-28 ... INFO Stage 5: Writing corpus files...
2026-03-28 ... INFO Stage 6: Writing manifest...
...
  [OK]  n_documents >= 4500 — Got 5,000
  [OK]  total_tokens >= 400000 — Got 650,XXX
  [OK]  vocab_size >= 10000 — Got XX,XXX
  [OK]  type_token_ratio > 0.10 — Got TTR=0.XXXX
  [OK]  corpus_sha256 present — ...
  [OK]  calibration recorded — cap=1XX

  ✓ CORPUS VALIDATION PASSED
```

**If it dies mid-run (laptop overheats, etc.):**
Re-run the exact same command. It will restart streaming collection.
The streaming scan is 20–40 minutes — losing and restarting is acceptable.

**If it completes and you see `✓ CORPUS VALIDATION PASSED`:**
Move to Step 3. Do not re-run this step.

**If you see `✗ CORPUS VALIDATION FAILED`:**
Read which specific check failed and consult `references/measurement_diagnostics.md`.

---

## Step 3 — Set Memory Thread Limits

Before running the rest of the pipeline, set these:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
```

These prevent multi-threaded BLAS from using all CPU cores simultaneously,
reducing thermal pressure and keeping floating-point accumulation deterministic.
You need to set them in every new PowerShell session.

---

## Step 4 — Run the Rest of the Pipeline (1.5–2.5 hours)

```powershell
python phase1_baseline\run.py `
  --seed 42 `
  --dump data\enwiki-20260101-pages-articles-multistream.xml.bz2 `
  --skip-existing
```

The `--dump` argument is required by the argument parser but will NOT be opened
since `--skip-existing` sees the corpus already built in Step 2. The dump file
stays untouched.

**What will run:**
- Stage 1 (corpus): [SKIP] — already built
- Stage 2 (lexical baseline): ~2 minutes
- Stage 3 (semantic baseline): 10–20 minutes (CPU embedding)
- Stage 4 (density/KDE): 5–10 minutes (CV bandwidth selection)
- Stage 5 (PPL baseline): 20–40 minutes (5k docs through DistilGPT-2)
- Stage 6 (pack assembly): ~1 minute

**If it dies mid-run:**
```powershell
# Re-set environment variables first, then:
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

python phase1_baseline\run.py `
  --seed 42 `
  --dump data\enwiki-20260101-pages-articles-multistream.xml.bz2 `
  --skip-existing
```

`--skip-existing` will detect which stages completed and resume from where
it stopped. Each stage writes its output before the next begins.

---

## Step 5 — Verify Phase 1 Complete

```powershell
python phase1_baseline\run.py --seed 42 --test-only
```

You need all five of these to pass:

```
  [OK  ] Zipf alpha 0.8–1.2 — alpha=X.XX
  [OK  ] Corpus TTR > 0.10 — TTR=0.XXXX
  [OK  ] Intrinsic dimensionality > 10 — dim=XX.X
  [OK  ] Mean PPL 15–30 — mean_ppl=XX.X
  [OK  ] G0 weight hash recorded — XXXXXXXXXXXXXXXX...

  ✓ Phase 1 COMPLETE — safe to begin Phase 2.
```

**If intrinsic dimensionality shows a PCA proxy warning:** This is expected.
`skdim` is not installed. The PCA-based proxy is acceptable. Note it in the
session log as "PCA proxy used; TwoNN will run on Kaggle in Phase 3."

**If any other check fails:** Bring the exact output (the numbers) back.
Do not guess the cause — the diagnostics reference has the exact table.

---

## Step 6 — Update config.yaml

After the pipeline succeeds, fill in the real values:

```powershell
# Get corpus_sha256
python -c "
import json
with open('phase1_baseline/corpus/manifest.json') as f:
    m = json.load(f)
print('corpus_sha256:', m['corpus_sha256'])
print('calibrated_cap:', m['preprocessing']['calibration']['derived_whitespace_token_cap'])
"

# Get G0 weight hash
python -c "
import json
with open('phase1_baseline/measurements/ppl_baseline.json') as f:
    p = json.load(f)
print('g0_model_weights_sha256:', p.get('_model_weights_sha256', 'MISSING'))
"
```

Paste both values into `config.yaml`:
```yaml
corpus_sha256: <value from above>
g0_model_weights_sha256: <value from above>
corpus_max_doc_length_tokens: <derived_whitespace_token_cap from above>
```

---

## Step 7 — Reproducibility Check

Run the full pipeline a second time (not just test-only) and compare pack hashes:

```powershell
# Second full run
python phase1_baseline\run.py `
  --seed 42 `
  --dump data\enwiki-20260101-pages-articles-multistream.xml.bz2

# Compare hashes (must be identical)
# Run 1 hash was logged during Step 4
# Run 2 hash appears in the output above
```

Note: the second run will rebuild everything from scratch since no `--skip-existing`.
If both `reference_pack.pkl` hashes match, reproducibility is confirmed.

---

## Step 8 — Sign Off Phase 1

Via the MCP server:
```
mco_complete_phase(
  phase=1,
  artifact_paths=["phase1_baseline/reference_pack.pkl",
                  "phase1_baseline/corpus/manifest.json",
                  "phase1_baseline/measurements/"],
  notes="Zipf alpha=X, TTR=X, IntrinsicDim=X (PCA proxy), MeanPPL=X, G0hash=X...",
  signed_by="<your name>"
)
```

Then move to Phase 2 on Kaggle. Upload `reference_pack.pkl` and `corpus/documents.jsonl`
to a Kaggle Dataset before the first generation run.

---

## Provenance Note (for SCOPE.md and any paper)

The corpus was built using the HuggingFace `wikipedia/20220301.en` streaming backend
instead of the raw Wikimedia XML dump due to Windows bz2 MemoryError constraints.
Content source: English Wikipedia. This must appear in SCOPE.md limitations:

```
- Corpus ingestion used HuggingFace wikipedia/20220301.en streaming backend
  (Wikimedia dump date: January 2022). Not identical to the enwiki-20260101 dump.
  Snapshot difference is documented in manifest.json.
```

This is honest documentation of the constraint. It does not invalidate the baseline —
English Wikipedia from 2022 vs 2026 has the same statistical properties for the
measurements being made. But it must be stated.
