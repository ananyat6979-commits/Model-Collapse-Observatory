"""
MCO Phase 3 — Kaggle Measurement Notebook
==========================================
Copy each cell block into a Kaggle notebook in order.
Attach mco-phase2-artifacts as input. GPU T4 x2. Internet ON.

Expected runtime: ~3-5 hours for all 4 generations.
Save Version after EACH generation (Cells 8, 9, 10, 11).
"""

# ── Cell 1: Imports and setup ──────────────────────────────────────
import os, json, pickle, math, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIR  = Path("/kaggle/input/datasets/ananyatiwari0212/mco-phase2-artifacts")
OUTPUT_DIR = Path("/kaggle/working/phase3_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print("Input:", sorted([f.name for f in INPUT_DIR.iterdir()]))


# ── Cell 2: Load reference pack and baseline files ─────────────────
with open(INPUT_DIR / "reference_pack.pkl", "rb") as f:
    pack = pickle.load(f)

print(f"encoder_id:     {pack.get('encoder_id')}")
print(f"tail_threshold: {pack.get('tail_threshold'):.4f}")
print(f"PCA shape:      {pack['embeddings_pca'].shape}")

# Load supplementary baseline files (upload these from laptop first)
for fname, key in [("lexical_baseline.json", "lexical"),
                    ("semantic_baseline.json", "semantic"),
                    ("ppl_baseline.json", "ppl_baseline"),
                    ("kl_baseline_distributions.json", "kl_distributions")]:
    p = INPUT_DIR / fname
    if p.exists():
        with open(p) as f:
            pack[key] = json.load(f)
        print(f"Loaded {fname}")
    else:
        print(f"MISSING: {fname} — upload to Kaggle dataset before running")
        pack[key] = {}


# ── Cell 3: Load frozen encoder ────────────────────────────────────
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(pack.get("encoder_id",
    "sentence-transformers/all-MiniLM-L6-v2"))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

assert not any(p.requires_grad for p in encoder.parameters()), \
    "CRITICAL: encoder not frozen"
print(f"Encoder frozen: {pack.get('encoder_id')}")


# ── Cell 4: Measurement functions ─────────────────────────────────

def measure_lexical(docs, pack):
    tokenized = [d.split() for d in docs]
    all_tokens = [t for toks in tokenized for t in toks]
    vocab = Counter(all_tokens)
    total = len(all_tokens)
    ttr = len(vocab) / total if total else 0

    def ngram_entropy(n):
        counts = Counter()
        for toks in tokenized:
            for i in range(len(toks)-n+1):
                counts[tuple(toks[i:i+n])] += 1
        s = sum(counts.values())
        return -sum((c/s)*math.log2(c/s) for c in counts.values()) if s else 0

    # KL(generated || baseline)
    kl_dists = pack.get("kl_distributions", {})
    uni_base = kl_dists.get("unigram_distribution", {})
    lp = float(kl_dists.get("laplace_alpha", 1.0))
    kl = 0.0
    if uni_base and total > 0:
        bt = sum(uni_base.values())
        vs = len(uni_base)
        def bp(w): return (uni_base.get(w,0)*bt + lp) / (bt + lp*(vs+1))
        for w, c in vocab.items():
            gp = c/total
            b = bp(w)
            if gp > 0 and b > 0:
                kl += gp * math.log2(gp/b)
        kl = max(0.0, kl)

    sorted_counts = sorted(vocab.values(), reverse=True)
    zipf_alpha = None
    if len(sorted_counts) >= 10:
        ranks = np.arange(1, len(sorted_counts)+1, dtype=float)
        freqs = np.array(sorted_counts, dtype=float)
        A = np.column_stack([np.ones_like(ranks), np.log(ranks)])
        res = np.linalg.lstsq(A, np.log(freqs+1e-9), rcond=None)
        zipf_alpha = float(-res[0][1])

    lex_b = pack.get("lexical", {})
    be = lex_b.get("entropy_1gram", None)
    e1g = ngram_entropy(1)
    return {
        "ttr": round(ttr, 6),
        "entropy_1gram": round(e1g, 6),
        "entropy_3gram": round(ngram_entropy(3), 6),
        "kl_div_1gram": round(kl, 6),
        "zipf_alpha": round(zipf_alpha, 6) if zipf_alpha else None,
        "baseline_entropy_1gram": be,
        "entropy_1gram_rel_change": round((e1g-be)/be, 6) if be else None,
    }


def measure_semantic(docs, pack, encoder, seed=42):
    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]
    gen_emb = encoder.encode(docs, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True).astype(np.float32)
    gen_pca = pca.transform(gen_emb)

    # Avg pairwise cosine distance (sampled)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(gen_emb), size=min(500, len(gen_emb)), replace=False)
    sub = gen_emb[idx]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms = np.where(norms==0, 1, norms)
    normed = sub / norms
    sim = normed @ normed.T
    upper = np.triu_indices(len(normed), k=1)
    cos_dist = float(np.mean(1 - sim[upper]))

    # Intrinsic dim proxy
    from sklearn.decomposition import PCA as _PCA
    nc = min(len(gen_emb), gen_emb.shape[1])
    _p = _PCA(n_components=nc, random_state=42)
    _p.fit(gen_emb)
    id_val = float(np.searchsorted(np.cumsum(_p.explained_variance_ratio_), 0.95)+1)

    # Coverage
    from sklearn.mixture import GaussianMixture
    nc2 = min(10, len(baseline_pca)//10)
    gmm = GaussianMixture(n_components=nc2, covariance_type="diag", random_state=42)
    gmm.fit(baseline_pca)
    covered = sum(
        1 for i in range(nc2)
        if np.any(np.linalg.norm(gen_pca - gmm.means_[i], axis=1) <
                  1.5 * float(np.sqrt(np.mean(gmm.covariances_[i]))))
    )
    coverage = covered / nc2 if nc2 > 0 else 0

    sem_b = pack.get("semantic", {})
    bc = sem_b.get("avg_pairwise_cosine_distance",
         sem_b.get("avg_cosine_distance", None))
    return {
        "avg_pairwise_cosine_dist": round(cos_dist, 6),
        "intrinsic_dimensionality": round(id_val, 4),
        "semantic_coverage": round(coverage, 6),
        "baseline_avg_cosine_dist": bc,
        "cosine_dist_rel_change": round((cos_dist-bc)/bc, 6) if bc else None,
    }


def measure_tail_mass(docs, pack, encoder):
    kde = pack["kde"]
    threshold = float(pack["tail_threshold"])
    pca = pack["pca"]
    baseline_pca = pack["embeddings_pca"]
    gen_emb = encoder.encode(docs, batch_size=64, show_progress_bar=False,
                              convert_to_numpy=True).astype(np.float32)
    gen_pca = pca.transform(gen_emb)
    gen_ll = kde.score_samples(gen_pca)
    base_ll = kde.score_samples(baseline_pca)
    tail_frac = float(np.mean(gen_ll < threshold))
    base_tail = float(np.mean(base_ll < threshold))
    return {
        "mean_log_likelihood": round(float(np.mean(gen_ll)), 6),
        "std_log_likelihood":  round(float(np.std(gen_ll)), 6),
        "tail_mass_fraction":  round(tail_frac, 6),
        "tail_threshold":      round(threshold, 6),
        "baseline_tail_fraction": round(base_tail, 6),
        "tail_fraction_rel_change": round((tail_frac-base_tail)/base_tail, 6)
            if base_tail > 0 else None,
    }


def measure_ppl_inversion(docs, pack, gk_ckpt_path, device=DEVICE, max_s=200):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(42)
    samples = [docs[i] for i in sorted(
        rng.choice(len(docs), size=min(max_s, len(docs)), replace=False).tolist())]

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    g0 = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    g0.eval()
    for p in g0.parameters(): p.requires_grad = False

    gk = AutoModelForCausalLM.from_pretrained(gk_ckpt_path).to(device)
    gk.eval()
    for p in gk.parameters(): p.requires_grad = False

    def get_ppls(model):
        ppls = []
        with torch.no_grad():
            for text in samples:
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=128).to(device)
                if enc.input_ids.size(1) < 2: continue
                out = model(**enc, labels=enc.input_ids.clone())
                ppl = math.exp(out.loss.item())
                if math.isfinite(ppl): ppls.append(ppl)
        return ppls

    print(f"  PPL inversion: {len(samples)} samples, loading G0 + Gk...")
    ppls_g0 = get_ppls(g0)
    ppls_gk = get_ppls(gk)

    valid = [(p0,pk) for p0,pk in zip(ppls_g0,ppls_gk)
             if math.isfinite(p0) and math.isfinite(pk) and pk>0]
    if not valid:
        return {"error": "no valid pairs", "perplexity_inversion_ratio": None}

    ratios = [p0/pk for p0,pk in valid]
    mr = float(np.mean(ratios))
    return {
        "ppl_under_g0": round(float(np.mean([v[0] for v in valid])), 4),
        "ppl_under_gk": round(float(np.mean([v[1] for v in valid])), 4),
        "perplexity_inversion_ratio": round(mr, 6),
        "std_ppl_inversion_ratio": round(float(np.std(ratios)), 6),
        "n_valid_samples": len(valid),
        "ratio_deviation_from_1": round(mr - 1.0, 6),
    }


# ── Cell 5: Load synthetic docs helper ────────────────────────────
def load_gen_docs(k, n=5000, seed=42):
    candidates = [
        INPUT_DIR / f"G{k}_outputs" / "synthetic.jsonl",
        INPUT_DIR / f"generations/G{k}/outputs/synthetic.jsonl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path: raise FileNotFoundError(f"G{k} synthetic.jsonl not found")
    docs = [json.loads(l)["text"] for l in open(path)]
    if len(docs) > n:
        idx = sorted(np.random.default_rng(seed).choice(
            len(docs), size=n, replace=False).tolist())
        docs = [docs[i] for i in idx]
    print(f"G{k}: {len(docs)} docs")
    return docs


def run_and_save(k, docs, gk_ckpt):
    t = time.time()
    result = {
        "generation_k": k,
        "n_documents": len(docs),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lexical":              measure_lexical(docs, pack),
        "semantic":             measure_semantic(docs, pack, encoder),
        "tail_mass":            measure_tail_mass(docs, pack, encoder),
        "perplexity_inversion": measure_ppl_inversion(docs, pack, gk_ckpt),
    }
    elapsed = (time.time() - t) / 60
    with open(OUTPUT_DIR / f"measurements_G{k}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nG{k} done in {elapsed:.1f} min")
    print(json.dumps({
        "lexical.ttr": result["lexical"]["ttr"],
        "lexical.kl": result["lexical"]["kl_div_1gram"],
        "semantic.cos_dist": result["semantic"]["avg_pairwise_cosine_dist"],
        "tail.fraction": result["tail_mass"]["tail_mass_fraction"],
        "ppl.ratio": result["perplexity_inversion"].get("perplexity_inversion_ratio"),
    }, indent=2))
    return result


# ── Cell 6: Run G0 measurements ────────────────────────────────────
g0_ckpt = str(INPUT_DIR / "G0_checkpoint")
r0 = run_and_save(0, load_gen_docs(0), g0_ckpt)
print(">>> SAVE VERSION NOW <<<")


# ── Cell 7: Run G1 measurements ────────────────────────────────────
g1_ckpt = str(INPUT_DIR / "G1_checkpoint")
r1 = run_and_save(1, load_gen_docs(1), g1_ckpt)
print(">>> SAVE VERSION NOW <<<")


# ── Cell 8: Run G2 measurements ────────────────────────────────────
g2_ckpt = str(INPUT_DIR / "G2_checkpoint")
r2 = run_and_save(2, load_gen_docs(2), g2_ckpt)
print(">>> SAVE VERSION NOW <<<")


# ── Cell 9: Run G3 measurements ────────────────────────────────────
g3_ckpt = str(INPUT_DIR / "G3_checkpoint")
r3 = run_and_save(3, load_gen_docs(3), g3_ckpt)


# ── Cell 10: Summary table ─────────────────────────────────────────
all_results = [r0, r1, r2, r3]
with open(OUTPUT_DIR / "all_measurements.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n── Measurement Summary ─────────────────────────────────────")
print(f"{'Gen':>4} | {'TTR':>7} | {'KL':>7} | {'CosDst':>7} | "
      f"{'TailFr':>7} | {'PPLRat':>7} | {'EntrRel%':>9}")
print("-" * 72)
for r in all_results:
    k   = r["generation_k"]
    lex = r.get("lexical", {})
    sem = r.get("semantic", {})
    tl  = r.get("tail_mass", {})
    ppl = r.get("perplexity_inversion", {})
    er  = lex.get("entropy_1gram_rel_change")
    tr  = tl.get("tail_fraction_rel_change")
    print(
        f"  G{k} | "
        f"{lex.get('ttr',0):.5f} | "
        f"{lex.get('kl_div_1gram',0):.5f} | "
        f"{sem.get('avg_pairwise_cosine_dist',0):.5f} | "
        f"{tl.get('tail_mass_fraction',0):.5f} | "
        f"{ppl.get('perplexity_inversion_ratio','N/A')} | "
        f"{f'{er*100:+.2f}%' if er else 'N/A':>9}"
    )

print("\n>>> FINAL SAVE VERSION — Phase 3 measurements complete <<<")
