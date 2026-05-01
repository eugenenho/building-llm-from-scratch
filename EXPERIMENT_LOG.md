# OWT Model Experiment Log

Documenting the iteration on the 100M-param transformer LM trained on OpenWebText for the CS336 leaderboard. Period: 2026-04-28 → 2026-04-30.

---

## Setup

### Architecture (baseline 100M tied)
- d_model 640, d_ff 1728, 12 layers, 10 heads, RoPE θ=10000
- context_length 256, vocab 32000, SwiGLU FFN, pre-norm RMSNorm
- ~100M params untied / ~80M tied (LM head shares matrix with embedding)

### Training defaults
- batch_size 256, steps 30000, warmup 500, cosine cycle = 30000
- AdamW: betas (0.9, 0.95), wd 0.1, eps 1e-8, max_l2_norm 1.0
- Mixed precision: bf16 autocast on CUDA, fp32 logits for loss
- `torch.compile` on, TF32 matmul on
- Hardware: H200 SXM (B200 not yet available)

### Token budget
30000 steps × 256 bs × 256 ctx ≈ **1.97B tokens** per full run

---

## Tools & infrastructure built before iteration

- YAML config + CLI override pattern (argparse with `argparse.SUPPRESS` to distinguish "not passed" from "passed default")
- WandB logging of: training_loss, ce_loss, z_loss_term, log_z_abs_mean, lr, grad_norm, tokens
- Validation: separate `val_loss` (full objective), `val_ce_loss` (apples-to-apples cross-run), `val_ppl` (always computed from CE only, with try/except OverflowError fallback)
- `wandb.run.summary["diverged"]` set on `not math.isfinite(loss)` for filterable run-level fields
- Architecture toggles via shared config dict propagated through every module: `use_rms_norm`, `norm_position`, `use_rope`, `ffn_type`, `qk_norm`, `wang_init`, `z_loss`, `z_loss_alpha`, `weight_tying`
- `gradient_clipping` returns total l2 norm (pre-clip) for logging
- Param count print at model creation

---

## Phase 1 — LR sweep on baseline (no architecture changes)

**Hypothesis:** TinyStories optimal LR (1e-2) won't transfer to OWT because the model is deeper (12 vs 4 layers) and dataset is harder.

| Run | lr_max | Outcome |
|---|---|---|
| baseline-1e3 | 1e-3 | Stable. **Best of sweep.** Final val ≈ 3.48 (75min H200) |
| lr-sweep-3e3 | 3e-3 | Trains but worse than 1e-3 |
| lr-sweep-1e2 | 1e-2 | Big grad norm spikes, near-divergent |
| lr-sweep-3e2 | 3e-2 | Diverged (kept as endpoint) |

**Diagnostic from `owt_100m_lr_sweep.png`:**
- 1e-3 (green) won decisively
- 1e-3 still descending at end of run → undertrained
- grad_norm gradually rising throughout → model still finding useful gradients
- 1e-2 instability suggested room for stability bundle to unlock higher LR

### Mental models clarified
- **LR optimum scales down with model depth/size.** TinyStories optimum doesn't transfer.
- **"Edge of stability" framing:** highest LR you can use without divergence is often near-optimal. Sweep should bracket it.
- **Divergence vs convergence vocabulary:** "divergent" = NaN/Inf or grad explosion; "converged" = loss has flattened. They're not opposites — a run can be neither.
- **`not math.isfinite()`** catches both NaN and ±Inf in one check.
- **WandB step-level vs `wandb.run.summary`:** use summary for filterable run-level fields like `diverged: True`.

---

## Phase 2 — Hardware throughput

H200 SXM ≈ 0.5–0.6× B200 throughput (mix of compute and bandwidth). 75 min H200 ≈ 38–45 min B200 (right at leaderboard budget).

Earlier conversation's "B200 ≈ 0.7-0.8×" estimate was wrong — corrected to ~2× speedup B200 over H200. Practical implication: tuning on H200 is fine; just adjust `steps` to fit 45-min B200 budget at leaderboard time.

---

## Phase 3 — Stability bundle (qk_norm + wang_init + z_loss)

**Hypothesis:** Stability fixes will unlock higher LR (1e-2, 3e-2 that previously diverged), giving faster convergence in fixed wall-clock budget.

### Bug found before any stability runs
RMSNorm gain `g` was being initialized via truncated normal with std `√(2/d_model)` — values near 0. Handout (p.17) specifies `g = 1`. Fixed to `nn.Parameter(torch.ones(d_model))` globally. This was a silent bug present in all previous runs; pre-norm robustness hid it.

### Implementations (all gated by config dict)

**QK-norm:** RMSNorm on Q and K per-head before attention dot product. Bounds attention-logit magnitude. Critical placement: norm BEFORE RoPE, not after (norm-then-rotate commutes correctly; one shared `rmsnorm_q` and `rmsnorm_k` for all heads, since RMSNorm normalizes over last dim).

**Wang init (1/√(2L)):** scale W_o (attention output proj) and W_2 (FFN output proj) inits by `1/√(2·num_layers)`. Only the residual-output projections get scaled — W_q/W_k/W_v/W_1/W_3 are intermediate, not residual contributions. Required passing `num_layers` through `transformer_block → mha/ffn`.

**Z-loss:** add `α·(log Z)²` to loss with α=1e-4. Critical: `log Z` is the *true* (unshifted) log-partition-function, not the shifted version from the numerically-stable CE computation. Code returns `(ce_loss, log_z)` from `cross_entropy`, where `log_z = max_values.squeeze(-1) + second_term` (re-adds the max that was subtracted for numerical stability).

### Key bug avoided in z-loss validation
Initial val implementation: `iter_loss = val_ce_loss + α·z_loss if z_loss else ce_loss` — the `else` branch referenced training `ce_loss` not validation `val_ce_loss`. Caught and fixed.

### Validation logging design (resolved)
Compute `ce_loss` and `z_loss_term` *always* (regardless of flag), log both. Backprop on `ce + α·z_loss` only when `z_loss=True`. Compute `val_ppl = exp(val_ce_loss)` — always CE-only — so perplexity stays apples-to-apples across runs with/without z-loss.

### Results

| Run | lr_max | warmup | val_ce final | vs baseline (3.388) |
|---|---|---|---|---|
| stability3-1e3 | 1e-3 | 500 | ~3.39 | ≈ baseline (no help) |
| stability3-3e3-warmup1000 | 3e-3 | 1000 | 3.40 | -0.07 worse than baseline at every step |
| stability3-1e2 | 1e-2 | 500 | ~3.7 | Much worse, smooth flatline (bad basin) |
| stability3-1e2-warmup3000 | 1e-2 | 3000 | similar | Still flatlining |

**Conclusion:** Bundle did not help this model at any tested LR. Three data points (matching at 1e-3, slightly worse at 3e-3, much worse at 1e-2) is enough evidence to abandon. Diagnostic metrics confirm bundle is *working as designed* (`log_z_abs_mean` decreases, no grad spikes), but the underlying optimization is not stability-bound.

### Mental models clarified
- **Stability fixes don't help at LRs that were already stable.** They expand the *workable* LR range, not improve at the existing optimum.
- **Step-0 loss should be ≈ ln(vocab_size) ≈ 10.4** for a uniformly-initialized softmax. Sanity check after any init change.
- **Log-Z drift:** softmax is translation-invariant in logit space, so CE doesn't constrain absolute logit scale. Without z-loss, `log Z` does a slow random walk; with z-loss, it sits at small equilibrium where CE pull balances quadratic restoring force.
- **Z-loss equilibrium is healthy at any plateau value** (not 0). Higher LR → bigger random kicks → larger equilibrium.
- **Warmup must scale with peak LR.** lr=1e-3 / 500 warmup ≈ stable; lr=1e-2 needs ~3000+ warmup to ramp safely. AdamW's m̂ bias correction is unstable in first ~1/(1-β₂) ≈ 20 steps regardless.
- **Smooth flatline ≠ instability** — it's "early bad steps parked the model in a bad basin, late LR too small to escape." Different failure mode from grad-spike divergence.

### Setbacks during this phase
- Misread WandB chart legends multiple times. Critical lesson: re-check legend before declaring win/loss.
- argparse `type=bool` silently broken (`bool("False")` returns True). Removed bool CLI overrides; flags now YAML-only.
- Two runs at lr=1e-2 with stability3 both flatlined — wasted compute that could have gone to testing 3e-3 first.

---

## Phase 4 — Weight tying (clean isolation test)

**Hypothesis:** Stability isn't the bottleneck; capacity/inductive bias might be. Weight tying frees ~20.5M params (`V × D = 32000 × 640`) and forces shared input/output token representation.

### Implementation
- `embedding.__init__` reads `config["weight_tying"]`. When True, std = `1/√d_model ≈ 0.04`; when False, std = 1 (handout default).
- `transformer_lm.__init__`: create `lm_head` first, then if tied: `self.lm_head.w = self.token_embedding.embed`.
- One wasted allocation accepted for code cleanness (the original linear weights are GC'd before optimizer creation).
- Verified: param count drops from 100,449,920 → 79,969,920 = exactly 20,480,000 = 32000 × 640.

### Init tradeoff
- Untied embedding init wants std ≈ 1 (so embedding rows have unit RMS).
- LM head wants std ≈ 1/√D (so logits don't blow up).
- Tied case: pick LM-head-friendly std (1/√D ≈ 0.04). Embedding side fine because RMSNorm before block 1 normalizes.

### Result

| Step | Untied baseline (lr=1e-3) | Tied (lr=1e-3) | Δ |
|---|---|---|---|
| 999 | 4.420 | 4.396 | -0.024 |
| 4999 | 3.809 | 3.764 | -0.045 |
| 9999 | 3.669 | 3.606 | -0.063 |
| 14999 | 3.570 | 3.508 | -0.062 |
| 19999 | 3.477 | 3.435 | -0.042 |
| 24999 | 3.419 | 3.364 | -0.055 |
| **29999** | **3.388** | **3.339** | **-0.049** |

**Tied wins by ~0.05 nats throughout** — solid, real, in the typical 0.05-0.15 range for tied embeddings at small scale. Wall-clock per step is identical (tying is a quality lever, not speed lever).

### Mental models clarified
- **"Capacity-bound" vs "tokens-bound":** capacity-bound = model too small to fit needed function (loss plateaus high, more params helps); tokens-bound = model big enough but undertrained (loss still descending, more tokens helps).
- **Weight tying's wins come from inductive bias and regularization, not speed.** Forward/backward FLOPs unchanged. Only saves ~80MB params + ~160MB optimizer state (memory).
- **Untied vs tied embedding init:** untied wants std=1 (unit-RMS rows); tied wants std=1/√D (bounded logits when used in matmul). Different jobs of the same matrix have different requirements.
- **Z-loss diagnostic on tied run:** `z_loss_term ~150-180` (high) is fine when z_loss=False — the metric is *purely informational*. Tied model achieved best val_ce_loss while having highest log_z drift, proving stability bundle was solving a non-problem.

---

## Phase 5 — Capacity reinvestment (depth sweep)

**Hypothesis:** With 20.5M params freed by tying, reinvest into depth. Deeper = more compositional capacity but slower per step.

### Param math (per layer ≈ 4.96M; non-block ≈ 40.96M for embed+lm_head+final_norm)

| num_layers | Untied | Tied |
|---|---|---|
| 10 | ~90.5M | ~70M |
| 12 | ~100.5M | ~80M |
| 14 | ~110.4M | ~89.5M |

### Run 1: 14 layers tied (lr=1e-3, all else identical)
**Result:** Comparable val_loss x time vs 12 layers. Not capacity-bound at 12.

### Run 2: 10 layers tied + 35000 steps (in progress)
**Setup:** Reduced layers, bumped steps proportionally to use roughly fixed wall-clock budget (10L is ~17% faster per step). Tests "tokens-bound" hypothesis: if smaller-trained-longer wins, the model was data-hungry, not capacity-hungry.

**Predicted outcomes:**
- 10L beats 12L → tokens-bound; consider going smaller still
- 10L matches 12L → either works; pick whichever is operationally simpler
- 10L loses → capacity floor between 10 and 12; stay at 12

### Mental models clarified
- **Depth vs width trade at small scale:** depth slightly favored for tied models (depth doesn't grow embed matrix). Width tends to want lower LR (rough rule: optimal LR ∝ 1/√d_model).
- **Compare on `val_ce_loss x time`, not `x step`** when changing model size. Step-time changes shift the meaning of "step" comparison.
- **`cosine_cycle_iters` should always equal `steps`** — schedule should reach lr_min exactly at last step.
- **Don't reduce/grow steps and change architecture in the same run** — too many variables. If swapping depth, keep steps; do separate "fixed wall-clock" follow-up only after the depth question is answered.

---

## Cross-cutting learnings

### Hyperparameter exploration order
**Before architecture changes:** rough LR sweep on stable baseline. Don't perfect it.
**Then architecture, one change at a time**, each with a small local LR sweep around it (0.5×, 1×, 2× current best).
**Save model size sweep for last** — efficiency wins from architecture (tying, parallel block) shift the size frontier.

### Sanity checks for any architecture change
1. Param count: print `sum(p.numel() for p in model.parameters())` before/after.
2. Step-0 loss ≈ ln(vocab_size). Catches init bugs.
3. Compare ce_loss curve at known-good LR against prior baseline. Should match within noise — improvement comes from *unlocking new LRs*, not improving at old LR.
4. Diagnostic metrics: `log_z_abs_mean`, attention entropy, activation magnitudes. Often show clear differences even when loss looks identical.
5. Overfit a single batch (handout's recommendation): loss should drive to ~0 in <200 steps if architecture is correct.

### Cross-run comparison hygiene
- Always log `ce_loss` separately from total `loss` so apples-to-apples comparison works regardless of which runs have z_loss on.
- `val_ppl = exp(val_ce_loss)` always; never `exp(val_loss + z_term)`.
- Look at `val_loss x time` for size-changing experiments, `val_loss x step` for hyperparameter-only experiments at fixed model size.

### Verification of weight tying specifically
- Param count drops by V × D.
- `model.lm_head.w.data_ptr() == model.token_embedding.embed.data_ptr()` returns True.
- After backward, `embed.grad` is non-zero (gradients flow from both uses).

### Bugs caught and lessons
- **RMSNorm `g` init was zero-ish, should be 1** (handout p.17). Silent bug in all early runs.
- **`type=bool` in argparse is broken** (`bool("False")` is True). Removed CLI bool args.
- **Validation loss `else` branch referenced training variable** instead of validation variable. Single-character oversight in a ternary.
- **z-loss math: `log_z` must re-add `max_values`** that was subtracted for numerical stability. Otherwise you regularize the wrong quantity.
- **Cosine schedule must end at last step.** `cosine_cycle_iters = steps` always; mismatching wastes the schedule.
- **`final_norm` must be `Identity()` for post-norm** (post-norm's last block already ends with norm). Without this, double-normalization at output.

---

## Where things stand

**Best result on H200:** 12 layers tied, lr=1e-3, no stability bundle → **val_ce_loss 3.339 at 30000 steps** (~75-80 min H200, ~38-45 min B200 estimated).

**Confirmed wins:**
- Weight tying (-0.05 nats vs untied baseline)
- RMSNorm `g=1` init fix (silent before)

**Confirmed no-help:** QK-norm, Wang init, z-loss, lr above 1e-3, longer warmup at lr=1e-3.

**Open questions:**
- Does 10L + 35k steps (tokens-bound test) beat 12L + 30k steps?
- Would 2e-3 / 1.5e-3 with weight tying beat 1e-3? (Possibly free 0.02-0.05 more nats — tied models sometimes prefer slightly higher LR because embedding sees gradients from both uses.)
- Parallel block (PaLM-style: MHA and FFN in parallel from same RMSNorm(x)) — 10-15% throughput win, untested.

**Things considered but skipped:**
- Optimizer swap (Muon/Sophia) — implementation risk too high for leaderboard timeline.
- MoE / MLA / GQA — inference wins, not 100M-pretraining wins.
- Sliding window attention — context is 256, full attention is cheap.

---

## Appendix: Configs used

All configs in `configs/owt/`. Naming: `gpu-{Mparams}m-{tag}-{lr}.yaml`.

Notable configs:
- `gpu-100m-baseline-1e3.yaml` — clean baseline at lr=1e-3
- `gpu-100m-lr-sweep-{1e2,3e2,3e3}.yaml` — Phase 1 sweep
- `gpu-100m-stability3-{1e3,1e2,3e3}*.yaml` — Phase 3 stability bundle (with optional warmup variants)
- `gpu-100m-wtying-1e3.yaml` — Phase 4 weight tying winner
- `gpu-110m-wtying-1e3.yaml` — 14L depth experiment
- `gpu-90m-wtying-1e3.yaml` — 10L depth experiment (in progress at time of writing)
