"""
MCO Phase 1 — Corpus Ingestion Pipeline
=========================================
Produces a deterministic 5k-document corpus from a Wikipedia XML dump.

Usage:
    python phase1_baseline/corpus/ingest.py \
        --dump /path/to/enwiki-20251201-pages-articles-multistream.xml.bz2 \
        --output-dir phase1_baseline/corpus

Validate an existing corpus without re-running ingestion:
    python phase1_baseline/corpus/ingest.py --validate-only \
        --output-dir phase1_baseline/corpus

What this does:
    Stage 1  — Extract and clean articles (preliminary 500-token cap)
    Stage 0  — Tokenizer calibration on random sample of cleaned candidates
               (derives the real whitespace-token cap; HARD FAIL if transformers missing)
    Stage 1b — Apply calibrated cap to all candidates
    Stage 2  — MinHash deduplication
    Stage 3  — Deterministic sampling to target document count
    Stage 4  — Compute corpus statistics
    Stage 5  — Write documents.jsonl
    Stage 6  — Write manifest.json

What this does NOT do:
    - Lowercase          (preserves case for Zipf + TTR accuracy)
    - Remove punctuation (punctuation is part of the natural distribution)
    - Lemmatize or stem  (would destroy tail vocabulary)
    These decisions are intentional — over-cleaning destroys the tail distribution.

Reproducibility contract:
    Two runs with the same --seed on the same --dump must produce the same
    documents.jsonl (verified by corpus_sha256 in manifest.json).
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_DOCS         = 5000
MIN_DOC_TOKENS      = 50
MAX_DOC_TOKENS      = 1000   # default; replaced by empirical calibration at runtime
DEDUP_THRESHOLD     = 0.85   # Jaccard similarity threshold for near-duplicate removal
MINHASH_PERMUTATIONS = 128
MINHASH_NGRAM_SIZE  = 5      # character n-gram shingle size


# ── Encoding sanity check ──────────────────────────────────────────────────────

def is_clean_utf8(text: str) -> bool:
    """
    Reject documents with encoding artifacts.

    Wikipedia dump parsers silently mangle non-ASCII characters in ways that
    corrupt n-gram distributions. This catches the common failure modes:
      - Unicode replacement characters (\ufffd) from broken encodings
      - Excessive control characters from malformed HTML entities
      - Non-English text (>30% non-ASCII characters)

    This filter is NECESSARY but NOT SUFFICIENT — it catches gross corruption.
    Silent corruption (accent stripping, entity non-decoding) requires the
    sentinel tests in test_sentinel.py to catch.
    """
    if "\ufffd" in text:
        return False
    control_chars = sum(
        1 for c in text if unicodedata.category(c) in ("Cc", "Cf") and c not in "\n\t"
    )
    if control_chars > len(text) * 0.01:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.30:
        return False
    return True


# ── Simple whitespace tokenizer ───────────────────────────────────────────────

def whitespace_tokenize(text: str) -> list[str]:
    """
    Whitespace tokenization preserving punctuation attached to words.

    We do NOT use a learned tokenizer here because:
      - Lexical metrics (TTR, entropy, Zipf) should be on natural word forms
      - The HuggingFace tokenizer is used separately for model inputs
      - Mixing the two would create inconsistencies across measurement layers
    """
    return text.split()


# ── Encoder-aware length calibration ─────────────────────────────────────────

def calibrate_max_doc_length(
    sample_docs: list[str],
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    encoder_max_subword_tokens: int = 256,
    safety_margin: int = 20,
    target_percentile: float = 95.0,
) -> dict:
    """
    Derive a safe whitespace-token cap from the actual tokenizer's subword counts.

    The reference encoder (all-MiniLM-L6-v2) silently truncates inputs to 256
    word-pieces. Whitespace tokens are NOT the same as subword tokens: Wikipedia
    prose averages ~1.3–1.5 subword tokens per whitespace token, but this varies
    significantly by article type (technical articles with proper nouns and
    chemical names can reach >2.0).

    This function derives the whitespace-token cap empirically from the actual
    corpus+tokenizer pair, not from a guessed round number. The result is recorded
    in the manifest for full reproducibility.

    Pipeline invariant:
        This function MUST succeed before the corpus is built. If 'transformers'
        is not installed, the pipeline HARD FAILS — a soft fallback would allow
        the encoder to silently truncate documents and corrupt the semantic baseline.

    Args:
        sample_docs: cleaned documents (typically 500, drawn from candidate pool)
        encoder_id: HuggingFace model ID (must match the reference encoder in config.yaml)
        encoder_max_subword_tokens: encoder's hard truncation limit (256 for MiniLM)
        safety_margin: subword tokens to reserve below the limit (default 20)
        target_percentile: percentile of subword counts to use for cap derivation

    Returns:
        dict with derived cap and full calibration diagnostics for the manifest

    Raises:
        RuntimeError: if 'transformers' is not installed (HARD FAIL — see invariant)
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    except ImportError:
        raise RuntimeError(
            "Tokenizer calibration requires 'transformers' to be installed.\n"
            f"  Install: pip install transformers==4.38.2\n"
            f"  Encoder: {encoder_id}\n\n"
            "Calibration cannot be skipped. The encoder silently truncates inputs "
            "longer than its max_seq_length. Without calibration, corpus_max_doc_length_tokens "
            "is an unverified guess that may allow encoder truncation to corrupt the "
            "semantic baseline — specifically the tail mass and semantic diversity "
            "measurements, which are the primary signals of distributional collapse.\n\n"
            "Fix your environment before running ingestion:\n"
            "  pip install -r requirements.txt"
        ) from None

    subword_counts   = []
    whitespace_counts = []

    for doc in sample_docs:
        ws_toks  = len(doc.split())
        # Encode without special tokens to get the raw subword count
        sub_toks = len(tokenizer.encode(doc, add_special_tokens=False))
        subword_counts.append(sub_toks)
        whitespace_counts.append(ws_toks)

    subword_arr   = np.array(subword_counts)
    whitespace_arr = np.array(whitespace_counts)

    # Per-document ratio: subword tokens / whitespace tokens
    # Computed per document (not aggregate) to capture variability correctly
    ratios = subword_arr / np.maximum(whitespace_arr, 1)
    p95_ratio = float(np.percentile(ratios, target_percentile))

    safe_subword_budget   = encoder_max_subword_tokens - safety_margin  # 236 for MiniLM
    derived_whitespace_cap = int(safe_subword_budget / p95_ratio)

    p95_subword = float(np.percentile(subword_arr, target_percentile))
    p99_subword = float(np.percentile(subword_arr, 99.0))

    calibration = {
        "encoder_id":                         encoder_id,
        "encoder_max_subword_tokens":         encoder_max_subword_tokens,
        "safety_margin":                      safety_margin,
        "sample_size":                        len(sample_docs),
        "p95_subwords_per_whitespace_token":  round(p95_ratio, 4),
        "p95_subword_token_count":            round(p95_subword, 1),
        "p99_subword_token_count":            round(p99_subword, 1),
        "safe_subword_budget":                safe_subword_budget,
        "derived_whitespace_token_cap":       derived_whitespace_cap,
        "note": (
            f"Derived corpus_max_doc_length_tokens={derived_whitespace_cap} so that "
            f"the {target_percentile}th-percentile document uses ≤{safe_subword_budget} "
            f"subword tokens, fitting within the encoder's {encoder_max_subword_tokens}-token window. "
            f"Sample drawn from the cleaned candidate pool (not dump front matter) "
            f"to avoid systematic underestimation of subword density."
        ),
    }

    log.info(f"Tokenizer calibration on {len(sample_docs)} docs:")
    log.info(f"  p95 subword/whitespace ratio: {p95_ratio:.3f}")
    log.info(f"  p95 subword token count:      {p95_subword:.0f}")
    log.info(f"  Derived whitespace cap:        {derived_whitespace_cap} tokens")
    if p99_subword > safe_subword_budget:
        log.warning(
            f"  p99 subword count ({p99_subword:.0f}) exceeds safe budget ({safe_subword_budget}). "
            f"~1% of documents may still experience mild encoder truncation. "
            f"Reduce safety_margin or target_percentile to tighten if this is unacceptable."
        )

    return calibration


# ── MinHash deduplication ─────────────────────────────────────────────────────

def get_shingles(text: str, n: int = MINHASH_NGRAM_SIZE) -> set[str]:
    """Character n-gram shingles for MinHash."""
    text_lower = text.lower()
    return {text_lower[i : i + n] for i in range(len(text_lower) - n + 1)}


def minhash_signature(shingles: set[str], permutations: int, seed: int) -> np.ndarray:
    """
    Compute a MinHash signature for a set of shingles.

    Uses the same hash functions (same seed → same (a,b) parameters) for every
    document — this is correct MinHash behavior. The seed produces the same
    function parameters across all calls, enabling valid Jaccard estimation.

    Design note: review comment suggesting this creates "correlated hashes" is
    incorrect. MinHash requires IDENTICAL hash functions applied to all documents.
    The same RNG seed → same (a_i, b_i) pairs → same hash function for each
    permutation i. This is the intended design.
    """
    if not shingles:
        return np.full(permutations, np.iinfo(np.int64).max, dtype=np.int64)

    sig = np.full(permutations, np.iinfo(np.int64).max, dtype=np.int64)
    shingle_hashes = np.array(
        [int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31) for s in shingles],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    for i in range(permutations):
        a, b   = rng.integers(1, 2**31, size=2)
        hashed = (a * shingle_hashes + b) % (2**31 - 1)
        sig[i] = hashed.min()
    return sig


def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float((sig1 == sig2).mean())


# ── Wikipedia dump processing ─────────────────────────────────────────────────

def iter_wikipedia_articles(dump_path: Path) -> Iterator[dict]:
    """
    Extract namespace-0 articles from a Wikipedia XML dump.

    Handles both raw .xml and .xml.bz2 formats.
    Yields dicts with keys: title, text, id, ns

    Namespace filtering uses the authoritative <ns> XML element.
    ns=0 is the main article namespace; all others (Talk, User, etc.) are skipped.
    Title-based heuristics are NOT used — they silently reject valid articles
    like "U.S.–China trade relations" and produce false rejection counts.
    """
    import bz2
    import xml.etree.ElementTree as ET

    log.info(f"Opening Wikipedia dump: {dump_path}")

    if dump_path.suffix == ".bz2":
        opener = bz2.open(dump_path, "rt", encoding="utf-8")
    else:
        opener = open(dump_path, encoding="utf-8")

    xml_ns = "{http://www.mediawiki.org/xml/DTD/Special Export/}"

    with opener as f:
        current: dict = {}

        for event, elem in ET.iterparse(f, events=("start", "end")):
            tag = elem.tag.replace(xml_ns, "")

            if event == "start" and tag == "page":
                current = {}

            elif event == "end" and tag == "title":
                current["title"] = elem.text or ""

            elif event == "end" and tag == "ns":
                # Authoritative namespace check. 0 = main articles.
                current["ns"] = elem.text or "0"

            elif event == "end" and tag == "text":
                current["text"] = elem.text or ""

            elif event == "end" and tag == "id" and "id" not in current:
                current["id"] = elem.text or ""

            elif event == "end" and tag == "page":
                if (current.get("ns", "0") == "0"
                        and current.get("title")
                        and current.get("text")):
                    yield current
                elem.clear()  # release memory — critical for 22GB dumps


def clean_wikipedia_text(raw: str) -> str:
    """
    Strip Wikipedia markup while preserving natural language statistics.

    Conservative cleaning: removes markup syntax but keeps punctuation,
    case, and rare vocabulary. Over-cleaning destroys the tail distribution
    that the tail mass measurement is designed to detect.

    ── CLEANING ORDER MATTERS. DO NOT REARRANGE. ──────────────────────────────

    Step 1: Strip wiki templates {{...}}
            Must happen before HTML entity decoding. Templates can contain
            raw &amp; that we don't want decoded into ambiguous & characters
            before the template body is stripped.

    Step 2: Strip HTML markup tags <ref>, <br/>, etc.
            Must happen BEFORE html.unescape(). unescape() turns &lt; → <
            and &gt; → >, so running tag stripping after would destroy literal
            < and > in article text (e.g., math expressions like &lt;100&gt;).

    Step 3: html.unescape() — &amp; → &, &lt; → <, &quot; → ", etc.
            Now safe because real HTML tags are already stripped.

    Step 4: NFKC normalization — collapse compatibility Unicode variants.
            é as two codepoints → one. Must happen after unescape so HTML
            entities (if any survive as &eacute;) are decoded first.

    Steps 5+: Wiki link stripping, whitespace normalization.
    ───────────────────────────────────────────────────────────────────────────
    """
    import html as _html

    if raw.strip().lower().startswith("#redirect"):
        return ""

    text = raw

    # Step 1: Remove wiki templates {{...}}
    depth, result, i = 0, [], 0
    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1; i += 2
        elif text[i : i + 2] == "}}":
            depth = max(0, depth - 1); i += 2
        elif depth == 0:
            result.append(text[i]); i += 1
        else:
            i += 1
    text = "".join(result)

    # Step 2: Strip HTML markup tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3: Decode HTML entities
    text = _html.unescape(text)

    # Step 4: NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Step 5: Wiki link resolution [[target|display]] → display, [[link]] → link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Remove section header markers (keep text)
    text = re.sub(r"={2,}([^=]+)={2,}", r"\1", text)

    # Remove file/image references
    text = re.sub(r"\[\[(?:File|Image|Media):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Corpus builder ────────────────────────────────────────────────────────────

def build_corpus(
    dump_path: Path,
    output_dir: Path,
    target_docs: int = TARGET_DOCS,
    min_tokens: int = MIN_DOC_TOKENS,
    max_tokens: int = MAX_DOC_TOKENS,
    dedup_threshold: float = DEDUP_THRESHOLD,
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    calibration_sample_size: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build a clean, deduplicated, deterministic corpus from a Wikipedia XML dump.

    The whitespace-token length cap is derived empirically from the actual
    encoder's tokenizer — not from a guessed constant. The calibration HARD FAILS
    if 'transformers' is not installed, preventing silent encoder truncation from
    corrupting the semantic baseline.

    Pipeline stages (in execution order):
        Stage 1  — Collect ~5× target candidates (preliminary 500-token cap)
        Stage 0  — Calibrate whitespace cap from tokenizer on random sample
        Stage 1b — Re-truncate all candidates to calibrated cap
        Stage 2  — MinHash deduplication
        Stage 3  — Deterministic sampling to target_docs
        Stage 4  — Corpus statistics
        Stage 5  — Write documents.jsonl
        Stage 6  — Write manifest.json

    Note on stage numbering:
        Stage 0 (calibration) is numbered 0 because it is conceptually a
        prerequisite configuration step, but it executes between Stages 1 and 1b
        because it requires a representative sample from the candidate pool.
        Stages are numbered by logical role, not execution order.

    Returns:
        dict with keys: documents (list[str]), manifest (dict), stats (dict),
                        corpus_file (Path), manifest_file (Path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    log.info("=== MCO Corpus Ingestion Pipeline ===")
    log.info(f"Target: {target_docs:,} documents, {min_tokens}–{max_tokens} tokens each")
    log.info(f"Encoder: {encoder_id}")

    # ── Stage 1: Extract and clean (preliminary 500-token cap) ─────────────────
    # We collect candidates under a generous cap before calibration because
    # calibration needs a representative sample from the cleaned candidate pool.
    # Sampling from the dump's front matter would bias toward shorter stub articles
    # that underrepresent subword inflation in longer technical articles.
    PRELIMINARY_CAP = 500

    log.info("Stage 1: Extracting and cleaning articles (preliminary cap)...")
    candidates    = []
    n_raw         = 0
    n_too_short   = 0
    n_too_long    = 0   # updated in Stage 1b after calibration
    n_encoding_fail = 0
    n_non_article = 0   # articles that produce empty text after cleaning

    for article in iter_wikipedia_articles(dump_path):
        n_raw += 1
        if n_raw % 10_000 == 0:
            log.info(f"  Processed {n_raw:,} raw articles, {len(candidates):,} candidates")

        # Namespace filtering is done inside iter_wikipedia_articles() via <ns>.
        # DO NOT add title-based heuristics here — they silently reject valid
        # articles (e.g., "U.S.–China trade relations") and pollute the rejection
        # accounting in the manifest.

        cleaned = clean_wikipedia_text(article["text"])
        if not cleaned:
            n_non_article += 1
            continue

        if not is_clean_utf8(cleaned):
            n_encoding_fail += 1
            continue

        tokens = whitespace_tokenize(cleaned)
        if len(tokens) < min_tokens:
            n_too_short += 1
            continue

        # Truncate to preliminary cap — will be re-applied in Stage 1b
        if len(tokens) > PRELIMINARY_CAP:
            tokens  = tokens[:PRELIMINARY_CAP]
            cleaned = " ".join(tokens)

        candidates.append({
            "id":       article["id"],
            "title":    article["title"],
            "text":     cleaned,
            "n_tokens": len(tokens),
        })

        if len(candidates) >= target_docs * 5:
            log.info(f"  Collected {len(candidates):,} candidates — stopping for calibration")
            break

    log.info(f"Stage 1 complete: {n_raw:,} raw → {len(candidates):,} candidates")
    log.info(
        f"  Rejected: {n_too_short:,} too short, "
        f"{n_encoding_fail:,} encoding errors, "
        f"{n_non_article:,} empty after cleaning"
    )

    # Hard fail: if even the uncapped preliminary pool is too small, the dump or
    # filters are wrong. A warning here allows a quietly underfilled baseline.
    if len(candidates) < target_docs:
        raise RuntimeError(
            f"Corpus underfilled BEFORE calibration: collected {len(candidates):,} "
            f"candidates but need {target_docs:,}.\n"
            f"Causes: dump too small, filters too strict, or early-stop triggered "
            f"(target_docs × 5 = {target_docs * 5:,}).\n"
            f"Rejected breakdown: too_short={n_too_short:,}, "
            f"encoding_fail={n_encoding_fail:,}, non_article={n_non_article:,}"
        )

    # ── Stage 0: Tokenizer calibration on representative sample ────────────────
    # Runs AFTER Stage 1 so the sample is from the cleaned candidate pool,
    # not from the dump's front matter.
    #
    # Residual bias note: calibration samples are truncated at PRELIMINARY_CAP=500
    # tokens, so very long articles with denser subword inflation in their later
    # sections may slightly underrepresent the true p95 ratio. The 20-token safety
    # margin and conservative rounding account for this. See calibrate_max_doc_length
    # docstring for the full analysis.
    log.info(
        f"Stage 0: Tokenizer calibration on random sample of "
        f"{min(calibration_sample_size, len(candidates))} candidates..."
    )

    n_for_calibration  = min(calibration_sample_size, len(candidates))
    cal_indices        = rng.choice(len(candidates), size=n_for_calibration, replace=False)
    calibration_samples = [candidates[i]["text"] for i in cal_indices]

    # calibrate_max_doc_length raises RuntimeError if transformers is not installed.
    # This is intentional — see its docstring for the invariant.
    calibration_result = calibrate_max_doc_length(
        calibration_samples,
        encoder_id=encoder_id,
    )
    max_tokens = calibration_result["derived_whitespace_token_cap"]
    log.info(f"  Calibrated max_doc_length: {MAX_DOC_TOKENS} → {max_tokens} tokens")
    log.info(f"  Final max_tokens: {max_tokens}")

    # ── Stage 1b: Apply calibrated cap ─────────────────────────────────────────
    log.info(f"Stage 1b: Applying calibrated cap ({max_tokens} tokens) to all candidates...")
    n_recapped = 0
    for doc in candidates:
        tokens = whitespace_tokenize(doc["text"])
        if len(tokens) > max_tokens:
            tokens         = tokens[:max_tokens]
            doc["text"]    = " ".join(tokens)
            doc["n_tokens"] = len(tokens)
            n_recapped += 1
    n_too_long = n_recapped
    if n_recapped:
        log.info(f"  Re-truncated {n_recapped:,} documents to calibrated cap")

    # ── Stage 2: MinHash deduplication ────────────────────────────────────────
    log.info("Stage 2: MinHash deduplication...")

    signatures = [
        minhash_signature(get_shingles(doc["text"]), MINHASH_PERMUTATIONS, seed)
        for doc in candidates
    ]

    kept_indices: list[int] = []
    kept_sigs: list[np.ndarray] = []
    for i, sig in enumerate(signatures):
        if not any(jaccard_estimate(sig, ks) >= dedup_threshold for ks in kept_sigs):
            kept_indices.append(i)
            kept_sigs.append(sig)

    n_before_dedup = len(candidates)
    candidates     = [candidates[i] for i in kept_indices]
    log.info(
        f"Dedup: {n_before_dedup:,} → {len(candidates):,} "
        f"(removed {n_before_dedup - len(candidates):,} near-duplicates)"
    )

    # Hard fail after dedup: a silently underfilled corpus corrupts every downstream
    # measurement. Raise rather than warn.
    if len(candidates) < target_docs:
        raise RuntimeError(
            f"Corpus underfilled AFTER dedup: {len(candidates):,} docs, need {target_docs:,}.\n"
            f"The candidate pool may be too homogeneous for this dedup threshold "
            f"({dedup_threshold}). Options:\n"
            f"  1. Lower dedup_threshold (e.g., 0.80) to keep more near-duplicates\n"
            f"  2. Reduce early-stop multiplier (currently 5×) to collect more candidates\n"
            f"  3. Verify the dump is a full Wikipedia dump, not a subset"
        )

    # ── Stage 3: Deterministic sampling to target ──────────────────────────────
    log.info("Stage 3: Deterministic sampling...")

    if len(candidates) > target_docs:
        candidates.sort(key=lambda d: d["id"])  # deterministic order before sampling
        indices    = rng.choice(len(candidates), size=target_docs, replace=False)
        indices.sort()
        candidates = [candidates[i] for i in indices]

    log.info(f"Final corpus: {len(candidates):,} documents")

    # ── Stage 4: Corpus statistics ─────────────────────────────────────────────
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
        "n_documents":        len(candidates),
        "total_tokens":       total_tokens,
        "vocab_size":         vocab_size,
        "type_token_ratio":   round(ttr, 4),
        "mean_doc_tokens":    round(float(np.mean(token_counts)), 1),
        "std_doc_tokens":     round(float(np.std(token_counts)), 1),
        "min_doc_tokens":     int(np.min(token_counts)),
        "max_doc_tokens":     int(np.max(token_counts)),
    }
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    # Sanity warnings (not fails — the researcher decides on downstream action)
    if ttr < 0.10:
        log.warning(f"TTR {ttr:.4f} < 0.10 — corpus may be too repetitive")
    if vocab_size < 10_000:
        log.warning(f"Vocab size {vocab_size:,} < 10k — tail distribution will be thin")

    # ── Stage 5: Write corpus files ────────────────────────────────────────────
    log.info("Stage 5: Writing corpus files...")

    corpus_file = output_dir / "documents.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in candidates:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    corpus_hash = sha256_file(corpus_file)

    # ── Stage 6: Write manifest ────────────────────────────────────────────────
    manifest = {
        "_corpus_hash":    corpus_hash,
        "_generated_at":   __import__("datetime").datetime.now().isoformat(),
        "_seed":           seed,
        "source":          "Wikipedia XML dump (English, namespace 0 only)",
        "note":            (
            "BookCorpus excluded — not yet integrated. "
            "SCOPE.md and config.yaml reflect this. "
            "Do not claim BookCorpus provenance until the ingestion is actually implemented."
        ),
        "dump_path":       str(dump_path),
        "corpus_sha256":   corpus_hash,
        "preprocessing":   {
            "namespace_filter":          "ns=0 via XML <ns> element (authoritative)",
            "encoding_filter":           "utf-8 clean, <1% control chars, <30% non-ASCII",
            "min_doc_tokens":            min_tokens,
            "max_doc_tokens":            max_tokens,
            "max_doc_tokens_derivation": "empirical_tokenizer_calibration",
            "calibration":               calibration_result,
            "dedup_method":              "minhash_greedy",
            "dedup_threshold":           dedup_threshold,
            "minhash_permutations":      MINHASH_PERMUTATIONS,
            "minhash_ngram_size":        MINHASH_NGRAM_SIZE,
            "case":                      "preserved",
            "punctuation":               "preserved",
            "tokenizer_for_lexical":     "whitespace_split",
        },
        "stats":           stats,
        "n_raw_rejected":  {
            "too_short":     n_too_short,
            "too_long_truncated": n_too_long,
            "encoding_fail": n_encoding_fail,
            "non_article":   n_non_article,
        },
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Corpus written to: {corpus_file}")
    log.info(f"Manifest written to: {manifest_file}")
    log.info(f"Corpus SHA-256: {corpus_hash}")

    return {
        "documents":     [d["text"] for d in candidates],
        "manifest":      manifest,
        "stats":         stats,
        "corpus_file":   corpus_file,
        "manifest_file": manifest_file,
    }


# ── Corpus loader (for downstream phases) ─────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[str]:
    """
    Load the preprocessed corpus from documents.jsonl.
    Verifies corpus hash against manifest before returning.
    Call this from Phase 2 and Phase 3 — never load documents.jsonl directly.
    """
    corpus_file   = corpus_dir / "documents.jsonl"
    manifest_file = corpus_dir / "manifest.json"

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_file}. Run ingest.py first."
        )

    with open(manifest_file) as f:
        manifest = json.load(f)

    expected_hash = manifest.get("corpus_sha256")
    actual_hash   = sha256_file(corpus_file)

    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Corpus hash mismatch! Expected {expected_hash[:16]}..., "
            f"got {actual_hash[:16]}... — corpus may have been modified or corrupted."
        )

    documents = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line)["text"])

    log.info(f"Loaded {len(documents):,} documents (hash verified)")
    return documents


# ── Deterministic subset sampler ──────────────────────────────────────────────

def sample_corpus(documents: list[str], n: int, seed: int = 42) -> list[str]:
    """
    Deterministic subset for fast local testing and validation runs.
    Uses np.random.default_rng for stability across numpy versions.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(documents), size=min(n, len(documents)), replace=False)
    indices.sort()
    return [documents[i] for i in indices]


# ── SHA-256 helper ─────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Corpus validation ──────────────────────────────────────────────────────────

def validate_corpus(corpus_dir: Path) -> bool:
    """
    Run post-ingestion validation checks against the manifest.
    Returns True if all checks pass.

    These checks verify quantity and provenance. Distributional quality
    (Zipf alpha, entropy range, intrinsic dimensionality) is validated
    by run.py's sanity checks after the full pipeline completes.
    """
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
         f"Got {stats.get('n_documents', 0):,} documents"),

        ("total_tokens >= 400000",
         stats.get("total_tokens", 0) >= 400_000,
         f"Got {stats.get('total_tokens', 0):,} tokens"),

        ("vocab_size >= 10000",
         stats.get("vocab_size", 0) >= 10_000,
         f"Got {stats.get('vocab_size', 0):,} types"),

        ("type_token_ratio > 0.10",
         stats.get("type_token_ratio", 0) > 0.10,
         f"Got TTR={stats.get('type_token_ratio', 0):.4f}"),

        ("corpus_sha256 present",
         bool(manifest.get("corpus_sha256")),
         "Hash missing from manifest"),

        ("calibration recorded",
         bool(manifest.get("preprocessing", {}).get("calibration", {})
                       .get("derived_whitespace_token_cap")),
         (f"cap={manifest.get('preprocessing',{}).get('calibration',{}).get('derived_whitespace_token_cap','MISSING')}")),
    ]

    for name, result, detail in checks:
        status = "[OK] " if result else "[FAIL]"
        print(f"  {status} {name} — {detail}")
        if not result:
            passed = False

    print()
    print("  ✓ CORPUS VALIDATION PASSED" if passed else "  ✗ CORPUS VALIDATION FAILED")
    return passed


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCO corpus ingestion pipeline")
    parser.add_argument("--dump", type=Path,
                        help="Path to Wikipedia XML dump (.xml or .xml.bz2)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--target-docs", type=int, default=TARGET_DOCS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-id", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Encoder ID for tokenizer calibration (must match config.yaml)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate an existing corpus without re-running ingestion")
    args = parser.parse_args()

    import random
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.validate_only:
        ok = validate_corpus(args.output_dir)
        sys.exit(0 if ok else 1)

    if not args.dump:
        print("Error: --dump is required. Use --validate-only to check an existing corpus.")
        sys.exit(1)
    if not args.dump.exists():
        print(f"Error: dump file not found: {args.dump}")
        sys.exit(1)

    result = build_corpus(
        dump_path=args.dump,
        output_dir=args.output_dir,
        target_docs=args.target_docs,
        encoder_id=args.encoder_id,
        seed=args.seed,
    )

    validate_corpus(args.output_dir)
