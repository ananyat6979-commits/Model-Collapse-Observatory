"""
MCO Phase 1 — Corpus Ingestion Sentinel Tests
===============================================
These tests catch silent data corruption that the encoding filter cannot see.

The canonical failure mode: a Wikipedia XML parser silently truncates or mangles
non-ASCII characters. The resulting text is still valid UTF-8 (no replacement
characters), so is_clean_utf8() passes, but the text is wrong. Examples:

  "Pokémon" → "Pokmon"       (accent stripped)
  "&amp;foo" → "&amp;foo"    (entity not decoded, becomes vocab noise)
  "naïve"   → "naive"        (diaeresis stripped)
  "café"    → "caf"           (truncation at non-ASCII)

Run this before processing any real dump:
    python phase1_baseline/corpus/test_sentinel.py

All tests must print PASS. Any FAIL is a blocker — do not proceed to full ingest.
"""

import sys
import unicodedata
from pathlib import Path

# Make package importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase1_baseline.corpus.ingest import clean_wikipedia_text, is_clean_utf8


def run_test(name: str, result: bool, detail: str = "") -> bool:
    status = "PASS" if result else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return result


def test_non_ascii_survival_through_cleaner():
    """
    Core sentinel: non-ASCII characters must survive clean_wikipedia_text() intact.

    If your XML parser is silently stripping accents or truncating at non-ASCII
    boundaries, these will fail even though is_clean_utf8() would pass on the
    corrupted output.
    """
    print("\n── Sentinel: Non-ASCII survival through cleaner ──────────────────")
    all_passed = True

    cases = [
        # (input_text, expected_substring, test_name)
        (
            "The Pokémon franchise began in 1996.",
            "Pokémon",
            "accented e survives (é U+00E9)"
        ),
        (
            "The café in [[Paris]] serves naïve tourists.",
            "café",
            "accent grave survives (é U+00E9, café)"
        ),
        (
            "The naïve approach fails here.",
            "naïve",
            "diaeresis survives (ï U+00EF)"
        ),
        (
            "Björk is an Icelandic singer.",
            "Björk",
            "o-umlaut survives (ö U+00F6)"
        ),
        (
            "The résumé was reviewed.",
            "résumé",
            "multiple accents survive in same word"
        ),
        (
            "El Niño affects weather patterns.",
            "Niño",
            "tilde-n survives (ñ U+00F1)"
        ),
        (
            "The cliché was overused.",
            "cliché",
            "trailing accent survives"
        ),
    ]

    for raw, expected, name in cases:
        cleaned = clean_wikipedia_text(raw)
        passed = expected in cleaned
        all_passed &= run_test(name, passed,
                               f"expected '{expected}' in '{cleaned[:60]}'")

    return all_passed


def test_html_entity_decoding():
    """
    HTML entities must be decoded. If &amp; survives into the corpus text,
    it becomes a vocabulary item that pollutes Zipf and n-gram distributions.
    """
    print("\n── Sentinel: HTML entity decoding ───────────────────────────────")
    all_passed = True

    cases = [
        ("Fish &amp; chips are popular.", "&", "amp entity decoded to &"),
        ("Score: &lt;100&gt;", "<", "lt/gt entities decoded"),
        ("Non&nbsp;breaking space", " ", "nbsp decoded to space"),
        ("Quote: &quot;hello&quot;", '"', "quot entity decoded"),
        # The entity form must NOT survive
        ("Fish &amp; chips", "&amp;", False,
         "&amp; must not appear literally in output"),
    ]

    for case in cases:
        if len(case) == 4 and case[2] is False:
            raw, forbidden, _, name = case
            cleaned = clean_wikipedia_text(raw)
            passed = forbidden not in cleaned
            all_passed &= run_test(name, passed,
                                   f"'{forbidden}' must not appear in '{cleaned[:60]}'")
        else:
            raw, expected, name = case
            cleaned = clean_wikipedia_text(raw)
            passed = expected in cleaned
            all_passed &= run_test(name, passed,
                                   f"expected '{expected}' in '{cleaned[:60]}'")

    return all_passed


def test_nfkc_normalization():
    """
    NFKC normalization must collapse compatibility variants.
    Without this, the "same" character in two normalization forms creates
    two different vocabulary items in the Zipf distribution.
    """
    print("\n── Sentinel: NFKC normalization ─────────────────────────────────")
    all_passed = True

    # é as precomposed (U+00E9) vs decomposed (e + U+0301)
    precomposed = "caf\u00e9"           # é as single codepoint
    decomposed  = "cafe\u0301"          # e + combining acute accent

    cleaned_pre  = clean_wikipedia_text(precomposed)
    cleaned_dec  = clean_wikipedia_text(decomposed)

    passed = cleaned_pre == cleaned_dec
    all_passed &= run_test(
        "precomposed é == decomposed é after normalization",
        passed,
        f"'{cleaned_pre}' == '{cleaned_dec}'"
    )

    # Verify both produce the NFKC form
    expected = unicodedata.normalize("NFKC", precomposed)
    passed = cleaned_pre == expected
    all_passed &= run_test(
        "output is NFKC form",
        passed,
        f"expected '{expected}', got '{cleaned_pre}'"
    )

    return all_passed


def test_encoding_filter_still_catches_corruption():
    """
    Verify that is_clean_utf8() still catches what it's supposed to catch,
    even after adding NFKC normalization (normalization must not break the filter).
    """
    print("\n── Sentinel: Encoding filter integrity ──────────────────────────")
    all_passed = True

    cases = [
        ("Normal English text with Pokémon.", True, "clean text passes"),
        ("Text with \ufffd replacement character.", False, "replacement char rejected"),
        ("Normal text.", True, "simple ASCII passes"),
        # Text that is >30% non-ASCII should be rejected
        ("α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ", False,
         "mostly non-ASCII rejected"),
    ]

    for text, expected, name in cases:
        result = is_clean_utf8(text)
        passed = result == expected
        all_passed &= run_test(name, passed,
                               f"is_clean_utf8() returned {result}, expected {expected}")

    return all_passed


def test_redirect_pages_excluded():
    """Redirect pages must produce empty string from the cleaner."""
    print("\n── Sentinel: Redirect page exclusion ────────────────────────────")
    all_passed = True

    cases = [
        ("#REDIRECT [[Main article]]", "uppercase REDIRECT"),
        ("#redirect [[other page]]", "lowercase redirect"),
        ("  #Redirect [[page]]  ", "redirect with leading whitespace"),
    ]

    for raw, name in cases:
        cleaned = clean_wikipedia_text(raw)
        passed = cleaned == ""
        all_passed &= run_test(name, passed, f"got '{cleaned[:40]}'")

    return all_passed


def main():
    print("MCO Corpus Ingestion — Sentinel Tests")
    print("=" * 60)
    print("These tests must ALL pass before running ingest on a real dump.")
    print("Any FAIL is a blocker. Do not proceed until all pass.")

    results = [
        test_non_ascii_survival_through_cleaner(),
        test_html_entity_decoding(),
        test_nfkc_normalization(),
        test_encoding_filter_still_catches_corruption(),
        test_redirect_pages_excluded(),
    ]

    print("\n" + "=" * 60)
    n_passed = sum(results)
    n_total = len(results)

    if all(results):
        print(f"✓ ALL SENTINEL TESTS PASSED ({n_passed}/{n_total})")
        print("  Safe to proceed with real Wikipedia dump.")
        sys.exit(0)
    else:
        failed = [i + 1 for i, r in enumerate(results) if not r]
        print(f"✗ {n_total - n_passed} SENTINEL TEST(S) FAILED: groups {failed}")
        print("  DO NOT run ingest on a real dump until all sentinels pass.")
        print("  Fix the cleaner or parser, then re-run this file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
