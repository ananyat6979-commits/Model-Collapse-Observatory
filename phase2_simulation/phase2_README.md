# Phase 2: Collapse Simulation

All generation artifacts (checkpoints and synthetic outputs) live on the
Kaggle Dataset `ananyatiwari0212/mco-phase2-artifacts`. They are NOT committed
to git — checkpoints are ~320MB each.

## Training Loss Progression

| Generation | Training Mix | Loss | Δ |
|---|---|---|---|
| G0 | Pretrained (no fine-tuning) | null | — |
| G1 | H(50%) + S0(50%), R=0.5 | 2.5996 | — |
| G2 | H(50%) + S1(50%), R=0.5 | 1.9892 | -0.610 |
| G3 | H(50%) + S2(50%), R=0.5 | 1.6592 | -0.330 |

Decreasing loss per generation = model increasingly fits the contaminated
distribution. Diminishing returns (G1→G2 drop > G2→G3 drop) confirmed.
This is early-regime collapse, consistent with theory.

## Artifact Inventory (on Kaggle Dataset)

| Artifact | Notes |
|---|---|
| `G0_checkpoint/` | Pretrained distilgpt2. READ-ONLY. Never overwrite. |
| `G0_outputs/synthetic.jsonl` | 5000 docs from G0 |
| `G1_checkpoint/` | Fine-tuned on H+S0, R=0.5, 3000 steps |
| `G1_outputs/synthetic.jsonl` | 5000 docs from G1 |
| `G2_checkpoint/` | Fine-tuned on H+S1, R=0.5, 3000 steps |
| `G2_outputs/synthetic.jsonl` | 5000 docs from G2 |
| `G3_checkpoint/` | Fine-tuned on H+S2, R=0.5, 3000 steps |
| `G3_outputs/synthetic.jsonl` | 5000 docs from G3 |

## Files Committed to Git

- `collapse_manifest.json` — generation metadata, loss values, training mix
- `README.md` — this file

## Kaggle Access

```bash
kaggle datasets download ananyatiwari0212/mco-phase2-artifacts \
  -p phase2_simulation/artifacts --unzip
```
