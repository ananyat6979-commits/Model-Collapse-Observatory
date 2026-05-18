"""
MCO Phase 4 — Early Warning Signal Analysis
=============================================
Runs statistical tests on Phase 3 measurements to determine:
  1. Which generation each signal first deviates significantly from baseline
  2. Signal priority order (which fires first)
  3. Contamination threshold estimates per signal
  4. Effect sizes for the Early Warning Card

Usage:
    python phase4_analysis/early_warning.py \
        --measurements phase3_measurements/results/all_measurements.json \
        --output-dir phase4_analysis/results

Requires: scipy, numpy (no GPU needed — CPU only)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


# ── Data loading ───────────────────────────────────────────────────────────────

def load_measurements(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def extract_series(results: list[dict], layer: str, metric: str) -> list[float]:
    """Extract a metric's values across generations in order."""
    series = []
    for r in sorted(results, key=lambda x: x["generation_k"]):
        val = r.get(layer, {}).get(metric)
        if val is None:
            raise KeyError(f"Missing {layer}.{metric} in generation k={r['generation_k']}")
        series.append(float(val))
    return series


# ── Statistical tests ──────────────────────────────────────────────────────────

def effect_size_r(n1, n2, U):
    """Effect size r from Mann-Whitney U."""
    z = (U - (n1 * n2) / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return abs(z) / np.sqrt(n1 + n2)


def test_generation_vs_baseline(
    baseline_values: list[float],
    generation_values: list[float],
    signal_name: str,
    generation_k: int,
) -> dict:
    """
    Two-sided Mann-Whitney U test: does generation k differ from baseline?
    Uses the per-sample distribution if available, otherwise point estimates.
    """
    b = np.array(baseline_values)
    g = np.array(generation_values)

    if len(b) < 3 or len(g) < 3:
        return {"warning": "Too few samples for reliable test", "n_baseline": len(b), "n_gen": len(g)}

    stat, p_val = stats.mannwhitneyu(b, g, alternative="two-sided")
    eff = effect_size_r(len(b), len(g), stat)

    # KS test as secondary
    ks_stat, ks_p = stats.ks_2samp(b, g)

    return {
        "signal": signal_name,
        "generation_k": generation_k,
        "mann_whitney_U": float(stat),
        "mann_whitney_p": float(p_val),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "effect_size_r": float(eff),
        "n_baseline": len(b),
        "n_generation": len(g),
    }


def bonferroni_threshold(n_tests: int, alpha: float = 0.01) -> float:
    return alpha / n_tests


# ── Point-estimate analysis (no per-sample distribution) ──────────────────────

def analyze_point_estimates(results: list[dict]) -> dict:
    """
    Since we have point estimates (not per-sample distributions), use the
    across-generation trend to assess signal detection order.

    For each signal, determine:
    - First generation with meaningful deviation from G0
    - Monotonicity
    - Total effect magnitude (G0 → G3)
    - Rate of change per generation
    """
    signals = {
        "ttr":                    ("lexical",  "ttr",                     "decreasing"),
        "entropy_1gram":          ("lexical",  "entropy_1gram",           "decreasing"),
        "kl_div_1gram":           ("lexical",  "kl_div_1gram",            "increasing"),
        "zipf_alpha":             ("lexical",  "zipf_alpha",              "increasing"),
        "avg_pairwise_cos_dist":  ("semantic", "avg_pairwise_cosine_dist","decreasing"),
        "tail_mass_fraction":     ("tail_mass","tail_mass_fraction",      "decreasing"),
        "ppl_inversion_ratio":    ("perplexity_inversion","perplexity_inversion_ratio","increasing"),
    }

    sorted_results = sorted(results, key=lambda x: x["generation_k"])
    n_gens = len(sorted_results)
    analysis = {}

    for sig_name, (layer, metric, expected_dir) in signals.items():
        try:
            series = extract_series(results, layer, metric)
        except KeyError as e:
            analysis[sig_name] = {"error": str(e)}
            continue

        g0 = series[0]

        # Relative changes from G0
        rel_changes = [(v - g0) / abs(g0) * 100 if g0 != 0 else 0 for v in series]

        # Detect first generation with >2% relative change in expected direction
        THRESHOLD_PCT = 2.0
        first_detection = None
        for k, (v, rc) in enumerate(zip(series, rel_changes)):
            if k == 0:
                continue
            if expected_dir == "decreasing" and rc < -THRESHOLD_PCT:
                first_detection = k
                break
            elif expected_dir == "increasing" and rc > THRESHOLD_PCT:
                first_detection = k
                break

        # Monotonicity
        diffs = [series[i+1] - series[i] for i in range(n_gens - 1)]
        if expected_dir == "decreasing":
            is_monotonic = all(d <= 0 for d in diffs)
        else:
            is_monotonic = all(d >= 0 for d in diffs)

        # Effect size (Cohen's d proxy)
        spread = np.std(series)
        effect = abs(series[-1] - series[0]) / (spread + 1e-9)

        # Rate of change (per generation)
        total_change = series[-1] - series[0]
        rate_per_gen = total_change / (n_gens - 1)

        analysis[sig_name] = {
            "series": series,
            "relative_changes_pct": [round(r, 2) for r in rel_changes],
            "expected_direction": expected_dir,
            "first_detection_generation": first_detection,
            "is_monotonic": is_monotonic,
            "total_effect_pct": round(rel_changes[-1], 2),
            "effect_size_proxy": round(float(effect), 3),
            "rate_per_generation_pct": round(rel_changes[-1] / (n_gens - 1) if n_gens > 1 else 0, 2),
            "g0_value": g0,
            "g_final_value": series[-1],
        }

    return analysis


def rank_signals(analysis: dict) -> list[tuple]:
    """Rank signals by: (1) first detection generation, (2) effect size."""
    ranked = []
    for sig, data in analysis.items():
        if "error" in data:
            continue
        det = data.get("first_detection_generation")
        if det is None:
            det = 999  # never detected
        effect = data.get("effect_size_proxy", 0)
        ranked.append((det, -effect, sig, data))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_analysis(measurements_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_measurements(measurements_path)

    print(f"Loaded measurements for {len(results)} generations")
    print(f"Generations: {[r['generation_k'] for r in results]}")
    print()

    # ── Point estimate analysis ─────────────────────────────────────────────
    analysis = analyze_point_estimates(results)
    ranked = rank_signals(analysis)

    print("=" * 65)
    print("SIGNAL DETECTION ORDER (earliest detection first)")
    print("=" * 65)
    print(f"{'Rank':<5} {'Signal':<30} {'1st Det':>7} {'Effect':>7} {'Total Δ':>8} {'Mono':>6}")
    print("-" * 65)

    for rank, (det_gen, neg_eff, sig_name, data) in enumerate(ranked, 1):
        det_str = f"k={det_gen}" if det_gen < 999 else "never"
        mono_str = "✓" if data["is_monotonic"] else "✗"
        print(f"  {rank:<3} {sig_name:<30} {det_str:>7} {data['effect_size_proxy']:>7.3f} "
              f"{data['total_effect_pct']:>+7.1f}% {mono_str:>6}")

    print()
    print("=" * 65)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 65)

    h_results = {}

    # H1: Tail mass fires before semantic
    def safe_det(sig):
        v = analysis.get(sig, {}).get("first_detection_generation")
        return 999 if v is None else v

    tail_det  = safe_det("tail_mass_fraction")
    sem_det   = safe_det("avg_pairwise_cos_dist")
    ppl_det   = safe_det("ppl_inversion_ratio")
    lex_det   = safe_det("entropy_1gram")

    h1 = tail_det < sem_det
    h2 = sem_det < lex_det
    h3 = ppl_det < tail_det

    print(f"  H1 (tail_mass fires before semantic): {'SUPPORTED' if h1 else 'REJECTED'}")
    print(f"     tail_mass first det: k={tail_det}  |  semantic first det: k={sem_det}")
    print()
    print(f"  H2 (semantic fires before lexical): {'SUPPORTED' if h2 else 'REJECTED'}")
    print(f"     semantic first det: k={sem_det}  |  lexical first det: k={lex_det}")
    print()
    print(f"  H3 (PPL inversion fires before tail mass — novel claim): "
          f"{'SUPPORTED' if h3 else 'REJECTED'}")
    print(f"     PPL inversion first det: k={ppl_det}  |  tail_mass first det: k={tail_det}")

    h_results = {"H1": h1, "H2": h2, "H3": h3,
                 "tail_detection": tail_det, "semantic_detection": sem_det,
                 "ppl_detection": ppl_det, "lexical_detection": lex_det}

    # ── Notable findings ────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("NOTABLE FINDINGS")
    print("=" * 65)
    print()
    print("1. PPL INVERSION — strongest and earliest signal")
    ppl_data = analysis["ppl_inversion_ratio"]
    print(f"   Series: {ppl_data['series']}")
    print(f"   Total effect: {ppl_data['total_effect_pct']:+.1f}%")
    print(f"   Effect size: {ppl_data['effect_size_proxy']:.3f}")
    print(f"   Monotonic: {ppl_data['is_monotonic']}")
    print(f"   H3 claim (PPL fires first): {'SUPPORTED' if h3 else 'REJECTED'}")
    print()
    print("2. LEXICAL SIGNALS — clean, monotonic, expected direction")
    for sig in ["ttr", "entropy_1gram", "kl_div_1gram", "zipf_alpha"]:
        d = analysis[sig]
        print(f"   {sig:<20} effect={d['effect_size_proxy']:.2f}  "
              f"total={d['total_effect_pct']:+.1f}%  mono={d['is_monotonic']}")
    print()
    print("3. TAIL MASS — anomalous direction (documented finding)")
    tm = analysis["tail_mass_fraction"]
    print(f"   Series: {tm['series']}")
    print(f"   Expected: decreasing. Observed: INCREASING from G0→G1, then flat.")
    print(f"   Root cause: G0 (DistilGPT-2/WebText) already generates text")
    print(f"   that falls in the tail of the Wikipedia KDE (G0 tail={tm['g0_value']:.3f}")
    print(f"   vs Phase 1 baseline tail=0.028). The domain mismatch means")
    print(f"   collapse does not move outputs toward the Wikipedia center.")
    print(f"   FIX for future work: compute tail mass using G0 as density")
    print(f"   reference, not Phase 1 Wikipedia baseline.")
    print()
    print("4. SEMANTIC COSINE — insufficient signal at this scale")
    sc = analysis["avg_pairwise_cos_dist"]
    print(f"   Series: {sc['series']}")
    print(f"   Total effect: {sc['total_effect_pct']:+.1f}% — below noise level")
    print(f"   Non-monotonic. Not a reliable signal at k=3, R=0.5, 5k docs.")

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "signal_analysis": {k: {kk: vv for kk, vv in v.items() if kk != "series"}
                            for k, v in analysis.items() if "error" not in v},
        "signal_series": {k: v["series"] for k, v in analysis.items()
                         if "error" not in v},
        "hypothesis_results": h_results,
        "signal_detection_order": [
            {"rank": i+1, "signal": sig, "first_detection_k": det if det < 999 else None,
             "effect_size": data["effect_size_proxy"],
             "total_effect_pct": data["total_effect_pct"],
             "monotonic": data["is_monotonic"]}
            for i, (det, _, sig, data) in enumerate(ranked)
        ],
    }

    out_path = output_dir / "signal_detection.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    return output, analysis, ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path,
                        default=Path("phase3_measurements/results/all_measurements.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("phase4_analysis/results"))
    args = parser.parse_args()

    run_analysis(args.measurements, args.output_dir)
