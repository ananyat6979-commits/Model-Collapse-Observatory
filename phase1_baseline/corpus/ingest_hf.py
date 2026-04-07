"""
MCO Phase 1 — Corpus Ingestion via HuggingFace Datasets
=========================================================
Drop-in replacement for the raw bz2 XML parsing path.

Use this when the raw Wikipedia XML dump causes MemoryError during bz2
decompression (common on Windows with limited RAM). Produces identical
output format: documents.jsonl + manifest.json.

The HuggingFace Wikipedia dataset is pre-parsed English Wikipedia,
pinned to a specific snapshot date for reproducibility.

Usage:
    python phase1_baseline/corpus/ingest_hf.py \
        --output-dir phase1_baseline/corpus \
        --seed 42

If interrupted, re-run the same command. The script checks whether
a valid corpus already exists before doing any work.

After this completes, run the rest of the pipeline:
    python phase1_baseline/run.py --seed 42 --skip-existing
"""

import argparse
import hashlib
import html as _html
import json
import logging
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Constants (must match ingest.py exactly) ──────────────────────────────────

TARGET_DOCS           = 5000
MIN_DOC_TOKENS        = 50
DEDUP_THRESHOLD       = 0.85
MINHASH_PERMUTATIONS  = 128
MINHASH_NGRAM_SIZE    = 5

# HuggingFace dataset — pinned for reproducibility. Never change after Phase 1.
HF_DATASET_NAME    = "wikipedia"
HF_DATASET_CONFIG  = "20220301.en"
HF_DATASET_SPLIT   = "train"


# ── Text utilities (identical to ingest.py) ───────────────────────────────────

def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def is_clean_utf8(text: str) -> bool:
    if "\ufffd" in text:
        return False
    control_chars = sum(
        1 for c in text
        if unicodedata.category(c) in ("Cc", "Cf") and c not in "\n\t"
    )
    if control_chars > len(text) * 0.01:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.30:
        return False
    return True


def clean_hf_text(text: str) -> str:
    """
    Clean HuggingFace Wikipedia article text.

    HF's Wikipedia dataset is already parsed from XML — no wiki markup remains.
    We still normalize Unicode, decode residual HTML entities, and collapse
    whitespace. Much simpler than the XML dump cleaner, but the same invariants
    hold: case preserved, punctuation preserved, no lemmatization.
    """
    if not text or text.strip().lower().startswith("#redirect"):
        return ""
    text = _html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── MinHash (identical to ingest.py) ─────────────────────────────────────────

def get_shingles(text: str, n: int = MINHASH_NGRAM_SIZE) -> set[str]:
    text_lower = text.lower()
    return {text_lower[i: i + n] for i in range(len(text_lower) - n + 1)}


def minhash_signature(shingles: set[str], permutations: int, seed: int) -> np.ndarray:
    if not shingles:
        return np.full(permutations, np.iinfo(np.int64).max, dtype=np.int64)
    sig = np.full(permutations, np.iinfo(np.int64).max, dtype=np.int64)
    shingle_hashes = np.array(
        [int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 31) for s in shingles],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    for i in range(permutations):
        a, b   = rng.integers(1, 2 ** 31, size=2)
        hashed = (a * shingle_hashes + b) % (2 ** 31 - 1)
        sig[i] = hashed.min()
    return sig


def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float((sig1 == sig2).mean())


# ── Tokenizer calibration (identical logic to ingest.py) ─────────────────────

def calibrate_max_doc_length(
    sample_docs: list[str],
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    encoder_max_subword_tokens: int = 256,
    safety_margin: int = 20,
    target_percentile: float = 95.0,
) -> dict:
    """
    Derive a safe whitespace-token cap from the actual tokenizer's subword counts.
    Identical logic to ingest.py. Hard-fails if transformers is not installed.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    except ImportError:
        raise RuntimeError(
            "Tokenizer calibration requires 'transformers'.\n"
            "Install: pip install transformers==4.38.2"
        ) from None

    subword_counts    = []
    whitespace_counts = []
    for doc in sample_docs:
        ws  = len(doc.split())
        sub = len(tokenizer.encode(doc, add_special_tokens=False))
        subword_counts.append(sub)
        whitespace_counts.append(ws)

    subword_arr    = np.array(subword_counts)
    whitespace_arr = np.array(whitespace_counts)
    ratios         = subword_arr / np.maximum(whitespace_arr, 1)
    p95_ratio      = float(np.percentile(ratios, target_percentile))
    safe_budget    = encoder_max_subword_tokens - safety_margin
    derived_cap    = int(safe_budget / p95_ratio)

    log.info(f"  Calibration: p95 ratio={p95_ratio:.3f}, derived cap={derived_cap} tokens")

    return {
        "encoder_id":                        encoder_id,
        "encoder_max_subword_tokens":        encoder_max_subword_tokens,
        "safety_margin":                     safety_margin,
        "sample_size":                       len(sample_docs),
        "p95_subwords_per_whitespace_token": round(p95_ratio, 4),
        "safe_subword_budget":               safe_budget,
        "derived_whitespace_token_cap":      derived_cap,
    }


# ── SHA-256 helper ─────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Idempotency check ─────────────────────────────────────────────────────────

def corpus_already_valid(output_dir: Path) -> bool:
    """
    Return True if a valid corpus already exists at output_dir.

    Checks: manifest.json exists, documents.jsonl exists,
    and the corpus_sha256 in the manifest matches the actual file hash.
    This prevents re-running the full streaming pass if the corpus is already built.
    """
    manifest_file = output_dir / "manifest.json"
    corpus_file   = output_dir / "documents.jsonl"

    if not manifest_file.exists() or not corpus_file.exists():
        return False

    try:
        with open(manifest_file) as f:
            manifest = json.load(f)
        expected_hash = manifest.get("corpus_sha256", "")
        if not expected_hash:
            return False
        actual_hash = sha256_file(corpus_file)
        return actual_hash == expected_hash
    except Exception:
        return False


# ── Main build function ────────────────────────────────────────────────────────

def build_corpus_hf(
    output_dir: Path,
    target_docs: int = TARGET_DOCS,
    min_tokens: int = MIN_DOC_TOKENS,
    dedup_threshold: float = DEDUP_THRESHOLD,
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    calibration_sample_size: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build the MCO baseline corpus using the HuggingFace Wikipedia dataset.

    Produces the same documents.jsonl + manifest.json as ingest.py.
    Use this when the raw bz2 XML path causes MemoryError on Windows.

    Idempotent: if a valid corpus already exists at output_dir, returns
    immediately without re-running the streaming pass.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Idempotency: skip if already built ────────────────────────────────────
    if corpus_already_valid(output_dir):
        log.info(f"Corpus already exists and hash is valid at {output_dir} — skipping build.")
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        documents = []
        with open(output_dir / "documents.jsonl", encoding="utf-8") as f:
            for line in f:
                documents.append(json.loads(line)["text"])
        return {
            "documents":     documents,
            "manifest":      manifest,
            "stats":         manifest.get("stats", {}),
            "corpus_file":   output_dir / "documents.jsonl",
            "manifest_file": output_dir / "manifest.json",
        }

    rng = np.random.default_rng(seed)

    log.info("=== MCO Corpus Ingestion (HuggingFace backend) ===")
    log.info(f"Dataset: {HF_DATASET_NAME} / {HF_DATASET_CONFIG} / {HF_DATASET_SPLIT}")
    log.info(f"Target:  {target_docs:,} documents, min {min_tokens} tokens each")

    # ── Load dataset in streaming mode ────────────────────────────────────────
    log.info("Loading Wikipedia dataset in streaming mode (no full RAM allocation)...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets not installed. pip install datasets==2.18.0"
        )

    dataset = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        split=HF_DATASET_SPLIT,
        streaming=True,
        trust_remote_code=True,
    )

    # Shuffle with a fixed seed so the sample is not alphabetically biased
    # (the HF dataset is sorted by page title by default).
    # buffer_size=10_000 means we sample from the first 10k articles at a time,
    # which is sufficient to remove alphabetical ordering without excessive RAM.
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    # ── Stage 1: Collect candidates with preliminary 500-token cap ────────────
    # We collect up to 5× the target docs before calibration. The preliminary
    # cap is generous (500 tokens) and will be replaced by the calibrated value.
    PRELIMINARY_CAP = 500

    log.info("Stage 1: Collecting and cleaning candidates (preliminary cap)...")
    candidates      = []
    n_raw           = 0
    n_too_short     = 0
    n_encoding_fail = 0
    n_empty         = 0

    for article in dataset:
        n_raw += 1
        if n_raw % 10_000 == 0:
            log.info(f"  Scanned {n_raw:,}, collected {len(candidates):,} candidates")

        text = article.get("text", "")
        if not text:
            n_empty += 1
            continue

        cleaned = clean_hf_text(text)
        if not cleaned:
            n_empty += 1
            continue

        if not is_clean_utf8(cleaned):
            n_encoding_fail += 1
            continue

        tokens = whitespace_tokenize(cleaned)
        if len(tokens) < min_tokens:
            n_too_short += 1
            continue

        # Apply preliminary cap — will be replaced in Stage 1b
        if len(tokens) > PRELIMINARY_CAP:
            tokens  = tokens[:PRELIMINARY_CAP]
            cleaned = " ".join(tokens)

        candidates.append({
            "id":       str(article.get("id", n_raw)),
            "title":    article.get("title", ""),
            "text":     cleaned,
            "n_tokens": len(tokens),
        })

        if len(candidates) >= target_docs * 5:
            log.info(f"  Collected {len(candidates):,} candidates — stopping for calibration")
            break

    log.info(f"Stage 1 complete: {n_raw:,} scanned → {len(candidates):,} candidates")
    log.info(
        f"  Rejected: {n_too_short:,} too short, "
        f"{n_encoding_fail:,} encoding errors, {n_empty:,} empty"
    )

    if len(candidates) < target_docs:
        raise RuntimeError(
            f"Corpus underfilled before calibration: {len(candidates):,} candidates, "
            f"need {target_docs:,}. The streaming scan may have been cut short."
        )

    # ── Stage 0: Tokenizer calibration on random sample from candidate pool ───
    # Runs AFTER Stage 1 so the sample is representative of the cleaned corpus,
    # not the first articles in the dataset (which would be alphabetically biased
    # even after shuffling if the buffer is small).
    log.info(f"Stage 0: Tokenizer calibration on random {calibration_sample_size} candidates...")
    n_cal       = min(calibration_sample_size, len(candidates))
    cal_idx     = rng.choice(len(candidates), size=n_cal, replace=False)
    cal_samples = [candidates[i]["text"] for i in cal_idx]

    calibration_result = calibrate_max_doc_length(cal_samples, encoder_id=encoder_id)
    max_tokens         = calibration_result["derived_whitespace_token_cap"]
    log.info(f"  Calibrated max_tokens: {max_tokens}")

    # ── Stage 1b: Apply calibrated cap ────────────────────────────────────────
    log.info(f"Stage 1b: Applying calibrated cap ({max_tokens} tokens)...")
    n_recapped = 0
    for doc in candidates:
        tokens = whitespace_tokenize(doc["text"])
        if len(tokens) > max_tokens:
            tokens          = tokens[:max_tokens]
            doc["text"]     = " ".join(tokens)
            doc["n_tokens"] = len(tokens)
            n_recapped += 1
    if n_recapped:
        log.info(f"  Re-truncated {n_recapped:,} documents to calibrated cap")

    # ── Stage 2: MinHash deduplication ────────────────────────────────────────
    log.info("Stage 2: MinHash deduplication (this may take a few minutes)...")
    signatures = [
        minhash_signature(get_shingles(doc["text"]), MINHASH_PERMUTATIONS, seed)
        for doc in candidates
    ]

    kept_indices: list[int] = []
    kept_sigs:    list[np.ndarray] = []
    for i, sig in enumerate(signatures):
        if not any(jaccard_estimate(sig, ks) >= dedup_threshold for ks in kept_sigs):
            kept_indices.append(i)
            kept_sigs.append(sig)

    n_before   = len(candidates)
    candidates = [candidates[i] for i in kept_indices]
    log.info(
        f"Dedup: {n_before:,} → {len(candidates):,} "
        f"(removed {n_before - len(candidates):,} near-duplicates)"
    )

    if len(candidates) < target_docs:
        raise RuntimeError(
            f"Corpus underfilled after dedup: {len(candidates):,} docs, need {target_docs:,}.\n"
            f"The candidate pool may be too homogeneous for dedup_threshold={dedup_threshold}."
        )

    # ── Stage 3: Deterministic sampling to target ─────────────────────────────
    log.info("Stage 3: Sampling to target document count...")
    if len(candidates) > target_docs:
        candidates.sort(key=lambda d: d["id"])
        indices    = rng.choice(len(candidates), size=target_docs, replace=False)
        indices.sort()
        candidates = [candidates[i] for i in indices]
    log.info(f"  Final corpus: {len(candidates):,} documents")

    # ── Stage 4: Corpus statistics ────────────────────────────────────────────
    log.info("Stage 4: Computing corpus statistics...")
    all_tokens   = []
    token_counts = []
    for doc in candidates:
        toks = whitespace_tokenize(doc["text"])
        all_tokens.extend(toks)
        token_counts.append(len(toks))

    total_tokens = len(all_tokens)
    vocab        = Counter(all_tokens)
    vocab_size   = len(vocab)
    ttr          = vocab_size / total_tokens if total_tokens > 0 else 0.0

    stats = {
        "n_documents":      len(candidates),
        "total_tokens":     total_tokens,
        "vocab_size":       vocab_size,
        "type_token_ratio": round(ttr, 4),
        "mean_doc_tokens":  round(float(np.mean(token_counts)), 1),
        "std_doc_tokens":   round(float(np.std(token_counts)), 1),
        "min_doc_tokens":   int(np.min(token_counts)),
        "max_doc_tokens":   int(np.max(token_counts)),
    }

    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    if ttr < 0.10:
        log.warning(f"TTR {ttr:.4f} < 0.10 — corpus may be too repetitive")
    if vocab_size < 10_000:
        log.warning(f"Vocab {vocab_size:,} < 10k — tail distribution will be thin")

    # ── Stage 5: Write corpus files ────────────────────────────────────────────
    log.info("Stage 5: Writing corpus files...")
    corpus_file = output_dir / "documents.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in candidates:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    corpus_hash = sha256_file(corpus_file)

    # ── Stage 6: Write manifest ────────────────────────────────────────────────
    manifest = {
        "_corpus_hash":  corpus_hash,
        "_generated_at": __import__("datetime").datetime.now().isoformat(),
        "_seed":         seed,
        "source":        "HuggingFace Wikipedia dataset (streaming)",
        "hf_dataset":    HF_DATASET_NAME,
        "hf_config":     HF_DATASET_CONFIG,
        "hf_split":      HF_DATASET_SPLIT,
        "note": (
            "Corpus built via HuggingFace datasets streaming backend. "
            "Raw bz2 XML dump path not used due to Windows bz2 MemoryError. "
            "Content source: English Wikipedia. Snapshot: 20220301.en. "
            "Ingestion path differs from ingest.py but output format is identical."
        ),
        "corpus_sha256": corpus_hash,
        "preprocessing": {
            "encoding_filter":             "utf-8 clean, <1% control chars, <30% non-ASCII",
            "min_doc_tokens":              min_tokens,
            "max_doc_tokens":              max_tokens,
            "max_doc_tokens_derivation":   "empirical_tokenizer_calibration",
            "calibration":                 calibration_result,
            "dedup_method":                "minhash_greedy",
            "dedup_threshold":             dedup_threshold,
            "minhash_permutations":        MINHASH_PERMUTATIONS,
            "minhash_ngram_size":          MINHASH_NGRAM_SIZE,
            "tokenizer":                   "whitespace_split (lexical metrics only)",
            "case":                        "preserved",
            "punctuation":                 "preserved",
        },
        "stats": stats,
        "n_raw_rejected": {
            "too_short":          n_too_short,
            "too_long_truncated": n_recapped,
            "encoding_fail":      n_encoding_fail,
            "empty":              n_empty,
        },
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Corpus written:   {corpus_file}")
    log.info(f"Manifest written: {manifest_file}")
    log.info(f"Corpus SHA-256:   {corpus_hash}")

    return {
        "documents":     [d["text"] for d in candidates],
        "manifest":      manifest,
        "stats":         stats,
        "corpus_file":   corpus_file,
        "manifest_file": manifest_file,
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_corpus(corpus_dir: Path) -> bool:
    print("\n── Corpus Validation ─────────────────────────────────────────────")
    manifest_file = corpus_dir / "manifest.json"
    corpus_file   = corpus_dir / "documents.jsonl"

    if not manifest_file.exists() or not corpus_file.exists():
        print("  [FAIL] Corpus files missing")
        return False

    with open(manifest_file) as f:
        manifest = json.load(f)

    stats  = manifest.get("stats", {})
    passed = True
    checks = [
        ("n_documents >= 4500",
         stats.get("n_documents", 0) >= 4500,
         f"Got {stats.get('n_documents', 0):,}"),

        ("total_tokens >= 400000",
         stats.get("total_tokens", 0) >= 400_000,
         f"Got {stats.get('total_tokens', 0):,}"),

        ("vocab_size >= 10000",
         stats.get("vocab_size", 0) >= 10_000,
         f"Got {stats.get('vocab_size', 0):,}"),

        ("type_token_ratio > 0.10",
         stats.get("type_token_ratio", 0) > 0.10,
         f"Got TTR={stats.get('type_token_ratio', 0):.4f}"),

        ("corpus_sha256 present",
         bool(manifest.get("corpus_sha256")),
         "Hash missing"),

        ("calibration recorded",
         bool(manifest.get("preprocessing", {}).get("calibration", {})
              .get("derived_whitespace_token_cap")),
         f"cap={manifest.get('preprocessing',{}).get('calibration',{}).get('derived_whitespace_token_cap','MISSING')}"),
    ]

    for name, result, detail in checks:
        print(f"  {'[OK] ' if result else '[FAIL]'} {name} — {detail}")
        if not result:
            passed = False

    print()
    print("  ✓ CORPUS VALIDATION PASSED" if passed else "  ✗ CORPUS VALIDATION FAILED")
    return passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO corpus ingestion via HuggingFace datasets")
    parser.add_argument("--output-dir",    type=Path, default=Path(__file__).parent)
    parser.add_argument("--target-docs",   type=int,  default=TARGET_DOCS)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--encoder-id",    type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate an existing corpus, do not rebuild")
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate_only:
        ok = validate_corpus(args.output_dir)
        sys.exit(0 if ok else 1)

    result = build_corpus_hf(
        output_dir=args.output_dir,
        target_docs=args.target_docs,
        encoder_id=args.encoder_id,
        seed=args.seed,
    )

    validate_corpus(args.output_dir)
