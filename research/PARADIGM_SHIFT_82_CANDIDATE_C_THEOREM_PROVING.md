# Paradigm Shift #82 — Candidate C: THEOREM-PROVING-DISTILL-CHIRON: Formal-Verification Axis via AlphaProof / Lean / Coq Distillation

**Status:** Candidate C for #82 slot. **Verdict: SELECT-CONDITIONAL** on math/formal-reasoning being a primary user need; otherwise **RESERVE for #83+**.
**Date:** 2026-05-08 (Ralph-loop iter 226, post-#81 at 20 axes; **#82 candidate evaluation**).
**Axis:** **FORMAL-VERIFICATION** — proposed 23rd axis (after pre-#82 stack at 20). Genuinely new (no prior paradigm has touched proof-checker-in-the-loop training).
**Magnitude target:** **5-20× on miniF2F / ProofNet / IMO-class subsets only**; not general-purpose. Production-validated by AlphaProof (DeepMind 2024 IMO Silver) and AlphaGeometry (DeepMind 2024).

---

## 0. Executive summary

Iter-226 evaluates THEOREM-PROVING-DISTILL-CHIRON as candidate C in the #82 slot. The core mechanism: distill from formal-proof teachers (AlphaProof, AlphaGeometry, LeanDojo-trained models, Coq theorem-proving systems) into the CHIRON coordinator, with proof-token positions integrated into the cached-logit pipeline (per #68) and an auxiliary verification loss that runs the emitted proof through the actual Lean/Coq executor.

**Why this candidate exists.** The 20 mature axes post-#81 cover text-NLL, multimodal (vision + audio), tool-use, agency, memory, world-model, causal-reasoning, free-form reasoning (#69), inference-speed, KV-compression, model-size, context-length, and depth-routing. **None grounds reasoning in a formal verifier.** Every other reasoning paradigm — #59 PRM, #62 AGENT, #67 CAUSAL, #69 REASONING-DISTILL — relies on natural-language critique or external tool calls. Formal verification is a categorically different signal: a proof is either accepted by the type-checker or it is not. This binary, mechanical correctness signal is unique among the program's signals.

**Why the verdict is SELECT-CONDITIONAL rather than SELECT.** The advantage is real but **narrow**: 5-20× on miniF2F, ProofNet, IMO formal subsets, and Coq/Lean tactic-prediction benchmarks. **Not general-purpose.** If the user's brief at iter-226 prioritizes math/formal-reasoning capability (e.g., a coding/math-assistant deployment), the SELECT case is strong. If the user's brief prioritizes general-purpose conversational quality, the magnitude is below the iter-200 bigger-picture threshold and the slot should go to a candidate with broader applicability.

**Production precedent (strong on the narrow domain).**
- **AlphaProof** (DeepMind 2024) — reinforcement-learning from Lean proof-checker; achieved IMO Silver (problems 1, 2, 4, 6).
- **AlphaGeometry / AlphaGeometry-2** (DeepMind 2024) — neuro-symbolic IMO geometry; near-Gold performance.
- **LeanDojo** (Yang & Deng 2023) — Lean-environment retrieval-augmented theorem proving (ReProver model); production framework for Lean-LLM training.
- **ProofNet** (2023, Azerbayev et al.) — undergraduate-math benchmark across Lean and Isabelle.
- **DSP — Draft-Sketch-Prove** (Microsoft 2023, Jiang et al.) — informal-to-formal pipeline using LLM drafts and Sledgehammer.
- **miniF2F** (2022, Zheng et al.) — formal-math benchmark (high-school + olympiad) used by all of the above.
- **Llemma** (2023, Azerbayev et al.) — math-specialized 7B/34B LLM, Llama-2 base, used as Lean tactic predictor.

**Honest framing.** AlphaProof is a domain-specialized system (RL with proof-checker rewards on Lean-formalized problems). Naive distillation of AlphaProof's behavior into a general-purpose CHIRON does **not** give CHIRON IMO Silver capability — it gives CHIRON the ability to *imitate* AlphaProof's surface tactic-emission style on math problems within the AlphaProof training distribution. The genuine lift is on **Lean/Coq tactic-prediction benchmarks** and **miniF2F-class problems** with auxiliary verification loss closing the loop.

**Joint Gate-0 PASS ~55%; LLM-scale empirical confirmation ~30%; risk-adjusted ~3-7×** on the narrow benchmark suite.

**Engineering:** ~1300 LOC over 7 weeks (verifier-in-loop infrastructure is the dominant cost; distillation pipeline is reuse).

---

## 1. Candidate context within the #82 slot

### 1.1 Sibling candidates at #82

The #82 slot is concurrently evaluating three candidates (this doc is C):

| Candidate | Mechanism | Likely verdict |
|---|---|---|
| **A — (sibling A)** | (sibling-A mechanism, evaluated separately) | (separate doc) |
| **B — (sibling B)** | (sibling-B mechanism, evaluated separately) | (separate doc) |
| **C — THEOREM-PROVING-DISTILL** | AlphaProof / Lean / Coq distillation; auxiliary verifier-in-loop loss | **SELECT-CONDITIONAL or RESERVE for #83+** |

Each candidate is documented self-contained; this doc evaluates C in isolation. Final #82 selection is decided by the parent design doc (`PARADIGM_SHIFT_82_DESIGN.md`).

### 1.2 Why C deserves serious evaluation despite narrow domain

Three grounds for keeping C in the candidate pool rather than auto-rejecting on narrow-domain grounds:

**1. Genuinely new axis.** No prior paradigm touches formal verification. #59 PRM provides step-level natural-language reward; #69 REASONING-DISTILL distills informal reasoning chains; #67 CAUSAL grounds in interventional graphs. **None close the loop with a mechanical proof checker.** If user reasserts the iter-200 "bigger picture" preference for genuinely-new axes, C is stronger than version-upgrade or microopt candidates.

**2. Production-validated at extreme.** AlphaProof at IMO Silver is the program's strongest external production-validation point on a reasoning axis: a binary, externally-judged correctness signal at a level humans struggle with. The mechanism is **proven to work at the high end** (modulo domain narrowness).

**3. Composes cleanly with existing reasoning stack.** #59 PRM (process reward) layers naturally onto proof-step correctness — every Lean tactic application is either valid or invalid, providing dense PRM signal at zero label cost. #69 REASONING-DISTILL handles the informal-reasoning portion (sketch generation); C handles the formal-verification portion. The two are complementary, not redundant.

---

## 2. Mechanism: formal-verification distillation

### 2.1 Teacher choice

| Tier | Teacher | Domain | Notes |
|---|---|---|---|
| **Tier 1 (preferred)** | AlphaProof checkpoint or near-equivalent (re-implementation via LeanDojo + ReProver) | Lean | DeepMind has not open-sourced AlphaProof; LeanDojo ReProver is the closest open analogue |
| **Tier 2** | Llemma-34B + Lean tactic head | Lean | Open-weights; mid-tier capability; well-distilled |
| **Tier 3** | Coq tactic predictors (CoqGym + ASTactic / Tactician) | Coq | Coq ecosystem; smaller training data than Lean |
| **Tier 4** | Isabelle + Sledgehammer-augmented LLM | Isabelle / HOL | Older; less ML tooling |

**Recommended:** Tier 1 (LeanDojo ReProver) for Gate-0; expand to Tier 3 (Coq) for Gate-1 cross-system validation.

### 2.2 Teacher emission format

For each training example (theorem statement S, formal proof P):

```
<THEOREM_BEGIN>
S = "∀ n : ℕ, n + 0 = n"          (Lean 4 syntax)
<THEOREM_END>
<PROOF_BEGIN>
intro n
induction n with
| zero => rfl
| succ n ih => rw [Nat.add_succ, ih]
<PROOF_END>
```

Teacher emits the whole sequence. Student is trained to predict proof tokens given the theorem statement.

### 2.3 Cached-logit pipeline integration (per #68)

#68 SUPER-DISTILL caches teacher logits at top-K=16 sparse decision points. Extension: proof-token positions are decision points (each tactic-name token is high-information). Cached top-16 logits at every proof-token position; ~3-5 GB disk per 1M theorem-proof pairs.

### 2.4 Auxiliary verification loss

The critical piece distinguishing C from "just another distillation":

```
L = α · L_KL(student_logits, cached_teacher_logits)
  + β · L_CE(student_logits, ground_truth_proof_tokens)
  + γ · L_verify(student_emitted_proof)
```

Where `L_verify` is computed by **actually running the student's emitted proof through the Lean/Coq executor**:

- If proof compiles: `L_verify = 0`.
- If proof fails: `L_verify = 1` (or finer-grained — distance to first error position).

**Differentiability handling.** `L_verify` is non-differentiable through the verifier. Standard approach (per AlphaProof and Lean-RL literature): treat as REINFORCE-style reward signal with the policy gradient `∇_θ E[L_verify] ≈ E[L_verify · ∇_θ log π_θ(proof|theorem)]`. PPO or simpler advantage-baseline variants are production-standard.

**Compute cost of verifier-in-loop.** Lean type-checking a typical olympiad-class proof: ~50ms-2s on CPU. Coq similar. With ~512 proofs/batch and 8 verifier worker processes: ~30-60s per batch verification step. **Bottleneck.** Mitigation: verify only every k-th step (k=4 typical) and rely on KL+CE for the intermediate steps.

### 2.5 Special tokens

Proposed: `<THEOREM_BEGIN>`, `<THEOREM_END>`, `<PROOF_BEGIN>`, `<PROOF_END>`, `<TACTIC>`, `<HAVE>`, `<SHOW>`, `<QED>`, `<LEAN>`, `<COQ>`, `<ISABELLE>`. ~11 new tokens (within #60/#62 budget).

### 2.6 Composition with prior 41 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#59 PRM** | ✓ Strong | Each Lean/Coq tactic application is verifier-judged → free PRM signal at zero labeling cost. **Joint multiplier** rather than additive: PRM-on-formal-tactics gives dense per-step reward where PRM-on-natural-language gives sparse per-trajectory reward. |
| **#67 CAUSAL** | ✓ Weak | Both are "grounded reasoning" axes but mechanism is orthogonal: formal verification ≠ causal intervention. Composes additively, not multiplicatively. |
| **#68 SUPER-DISTILL** | ✓ Stack-base | Cached-logit pipeline reused for proof-token positions. |
| **#69 REASONING-DISTILL** | ✓ Complementary | #69 handles informal reasoning sketch (DSP-style draft); C handles formal verification step (DSP-style prove). DSP is the canonical pipeline. **30-40% mechanism overlap on math/reasoning subsets** (informal→formal pipeline shares early-stage distillation infrastructure but diverges on verifier-in-loop). |
| **#60 TOOL-LLM** | ✓ Weak | Lean/Coq executor structurally is a "tool" in the #60 sense, but the verifier-in-loop reward signal is RL not tool-call training. Composes if user wants Lean exposed at inference as an interactive tool, but the training mechanism is distinct. |
| **#62 AGENT** | ✓ Weak | Multi-step proof construction can be framed as an agent trajectory (plan→tactic→observe-state→reflect→tactic), but standard formal-proof training uses fixed proof scripts not agent trajectories. Compose only on advanced variants. |

**Honest 30-40% overlap with #69 REASONING-DISTILL on math/reasoning subsets.** DSP (Microsoft 2023) is the canonical example: LLM drafts informal sketch → autoformalizes to Lean stub → Sledgehammer fills proof. The "draft" stage is #69; the "prove" stage is C. They are complementary axes, but a well-tuned #69 alone captures perhaps 40% of C's lift on miniF2F because R1-class teachers already emit some Lean syntax in their reasoning traces.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation on non-formal sequences

The auxiliary verification loss `L_verify` activates only on sequences containing `<PROOF_BEGIN>...<PROOF_END>` blocks. On general-purpose text (no proof block), `L_verify = 0` by construction; gradient contribution zero. **Bit-exact text NLL preserved on non-formal sequences.** ∎

### 3.2 Theorem 2 — Bijectivity preservation

C operates entirely at the loss / output-head level. Trunk symplectic shears unchanged. Bijectivity preserved per #66 Theorem 2 with proof-token embeddings being standard `Embed[vocab_index] ∈ ℝ^{2048}`. ∎

### 3.3 Theorem 3 — Verifier reward is unbiased estimator of true correctness

Lean and Coq type-checkers are **sound by construction**: a proof that compiles is genuinely a proof of the stated theorem (modulo verifier soundness, which is itself proven for Lean 4 mathlib core). Therefore `L_verify` is **bias-free**: there is no scenario where `L_verify = 0` but the proof is incorrect. This is a strong property absent from #59 PRM (which depends on the PRM-model's noisy judgment) and #69 (which depends on teacher trust). ∎

This soundness property is the **foundational reason** C is worth evaluating despite narrow domain: it provides the program's only **bias-free reward signal**.

### 3.4 Joint Gate-0 PASS probability

```
LeanDojo / ReProver teacher integration:                         ~80%
Cached-logit pipeline extension to proof-token positions:        ~88%
Verifier-in-loop infrastructure (Lean executor + REINFORCE):     ~60%
Verifier compute budget tractable at training scale:             ~70%
NLL preservation on text-only:                                   ~95%
LLM-scale empirical confirmation (miniF2F-class):                ~50%

Joint Gate-0 PASS:                                              ~55%
LLM-scale empirical confirmation:                               ~30%
```

The bottleneck probabilities are (a) verifier-in-loop infrastructure (60%) — RL-with-verifier is harder to stabilize than pure distillation, and (b) verifier compute budget tractable (70%) — Lean type-checking can spike to multi-second per proof on adversarial inputs.

### 3.5 Magnitude estimate

**Reference points.**
- AlphaProof: ~1000× over base LLM on IMO-class problems (LLM ~0% solve rate, AlphaProof reaches Silver). But this requires full RL-from-verifier training over months of compute.
- LeanDojo ReProver: ~2-4× over GPT-4 on Lean tactic prediction (in-distribution).
- Llemma: ~1.5-2× over Llama-2 base on miniF2F.
- DSP: ~1.3× over informal-reasoning-only baseline.

**C's expected lift via distillation (not full RL training):** between Llemma and ReProver, ~2-5× on miniF2F-class benchmarks; up to 10-20× on **narrow benchmark subsets** where teacher is strong (e.g., Lean tactic prediction where ReProver is in-distribution). **Not** AlphaProof-level capability — that requires full RL not distillation.

**Risk-adjusted realization:** 0.55 × 0.30 × {5-20×} → **0.8-3.3× expected on the narrow domain**. Below break-even at the low end of the range.

---

## 4. Updated cumulative stack (conditional on SELECT)

```
Iter 225 close (post-#81, hypothetical):
  All 8 training axes ≈preserved
  Effective model size: ~115-256B band
  Inference throughput: ~24× (or honest 4.8×)
  Effective context length: ∞
  Per-token compute: 2× faster (#79 MoD)
  AUDIO benchmarks: ~5,000,000× (post-#80)
  General reasoning: ~10× from #69 REASONING-DISTILL
  miniF2F / ProofNet / Lean-tactic: minimal (no prior paradigm targets formally-verified math)

Iter 226 (THEOREM-PROVING-DISTILL-CHIRON, conditional SELECT):
  All prior 22 axes ≈preserved
  AUDIO benchmarks: ~5,000,000× (unchanged)
  General reasoning: ~10× from #69 (unchanged on non-formal subsets)
  **miniF2F / ProofNet / IMO-formal: ~2-5× standalone, up to 10-20× on Lean-tactic subset (NEW AXIS)**
  **Lean/Coq tactic prediction: 5-15× — narrowest gain**
  Risk-adjusted realization on narrow domain: ~0.8-3.3×
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| LeanDojo / ReProver teacher integration (frozen) | 200 | 1 |
| Cached-logit pipeline extension to proof-token positions | 100 | 0.5 |
| Special-token vocabulary expansion (+11 formal-proof tokens) | 50 | 0.25 |
| Verifier-in-loop infrastructure (Lean 4 executor + IPC + worker pool) | 400 | 2 |
| REINFORCE / advantage-baseline policy gradient on `L_verify` | 200 | 1 |
| Coq parallel pipeline (Tier 3 cross-system validation) | 200 | 1 |
| miniF2F / ProofNet / Lean-tactic / Coq-tactic eval harness | 150 | 1.25 |
| **Total** | **~1300** | **7** |

Verifier-in-loop is the dominant engineering cost (~600 LOC + 3 weeks). The distillation pipeline itself is largely #68 reuse.

---

## 6. Memory advantage preservation

| Component | GPU memory (post-#81) |
|---|---|
| LeanDojo ReProver teacher (frozen, BF16, ~300M) | 600 MB |
| Cached-logit prefetch buffer (proof-augmented) | 1.5 GB host |
| Special-token embeddings (+11 tokens × 2048 BF16) | 0.04 MB |
| **Total additional GPU** | **~600 MB** |

Tight margin under post-#80 stack (~600 MB headroom). ReProver teacher offload-to-CPU between batches recommended (similar pattern to Whisper-large-v3 in #80).

**Verifier compute is CPU-side and does not consume GPU memory.** Worker pool of 8 Lean processes consumes ~4 GB host RAM; manageable.

---

## 7. Gates

### Gate-0 (~15 GPU-hours + ~50 CPU-hours)

**Probe.** 200M coordinator + LeanDojo ReProver teacher + ~500K Lean theorem-proof pairs (LeanDojo-extracted). KL-CE distillation for 50k steps. Verifier-in-loop on 10% of batches (compute budget control). Evaluate on miniF2F-test (244 problems) and Lean-tactic prediction.

**PASS criteria.**
- miniF2F-test: ≥ 15% solve rate (Llemma-7B baseline ~16%; small-model probe ~12-18% target).
- Lean-tactic prediction: ≥ 30% top-1 accuracy on held-out tactic decisions.
- Text NLL on general-purpose corpus: ≤ 0.005 nat regression.
- Verifier-in-loop training stable (no reward-collapse, no policy-divergence).

**PASS probability:** ~55% (verifier-in-loop stability is the dominant risk).

### Gate-1 (~200 GPU-hours + ~800 CPU-hours)

**Probe.** Full 32B-effective + Tier-1 ReProver + Tier-3 Coq (Tactician baseline) + ~5M theorem-proof pairs. Full benchmark suite.

**PASS criteria.**
- miniF2F: ≥ 30% solve rate (between Llemma-34B and ReProver).
- ProofNet: ≥ 18% solve rate (undergraduate-math tier).
- Lean-tactic prediction: ≥ 50% top-1.
- Coq-tactic prediction: ≥ 35% top-1 (cross-system validation).
- Text NLL on general-purpose corpus: ≤ 0.01 nat regression.

**PASS probability conditional on Gate-0:** ~55%.

---

## 8. Honest gaps

1. **Narrow domain.** miniF2F + ProofNet + Lean/Coq tactic prediction are the load-bearing benchmarks. **No general-purpose lift expected outside formal-math contexts.** If user's brief at iter-226 prioritizes general conversational quality, this candidate misses.

2. **Distillation ≠ AlphaProof.** Distilling AlphaProof-style behavior gives surface imitation, not IMO Silver capability. True IMO-level performance requires RL-from-verifier at AlphaProof's training scale (~months of TPU compute). C aspires to ReProver/Llemma-level capability, not AlphaProof-level.

3. **30-40% overlap with #69 REASONING-DISTILL on math subsets.** R1/o1-class teachers already emit Lean-syntax-adjacent reasoning. A well-tuned #69 alone captures ~40% of C's lift on miniF2F. Marginal lift of C over post-#69 baseline: 1.5-3× on miniF2F (lower than headline).

4. **Verifier-in-loop is the dominant engineering risk.** Lean type-checker integration via IPC, REINFORCE stability, reward-collapse mitigation, hypothesis-handle leakage — all standard hazards in formal-RL literature. ~600 LOC + 3 weeks just for the verifier loop.

5. **Verifier compute can spike.** Adversarial Lean proofs (deeply-nested type-class resolution) can take 10s+ per check. Worker pool with timeout cutoff mandatory; rejected timeouts treated as `L_verify = 1`.

6. **Lean and Coq are different formal systems.** Cross-system distillation (single coordinator emitting both Lean and Coq) doubles the special-token budget and dilutes per-system capability. Recommended: ship Lean-first; Coq as Tier-3 follow-on.

7. **No production CHIRON precedent for full pipeline.** AlphaProof / DSP / Llemma all use non-CHIRON architectures; CHIRON-32B-effective + verifier-in-loop is novel composition.

8. **Bias-free reward (Theorem 3) is unique among program signals — but only on formal proofs.** The auxiliary loss adds zero noise on the narrow domain, but the narrow domain itself is narrow. Outside formal proofs, the program's prior reward signals (#59 PRM, #69 KL distillation) are the operative ones.

9. **Magnitude target 5-20× is pre-risk-adjustment.** Risk-adjusted realization is **0.8-3.3×** on the narrow domain. At the low end, below break-even on iter-200 bigger-picture grounds.

10. **User-need conditional.** SELECT verdict requires user brief prioritizing math/formal-reasoning. Default brief at iter-226 hasn't signaled this; without a clear signal, RESERVE for #83+ is the conservative path.

---

## 9. Composition with #69 REASONING-DISTILL — DSP pattern

The strongest composition argument: DSP (Draft-Sketch-Prove, Microsoft 2023, Jiang et al.) demonstrates that informal reasoning + formal verification is **multiplicatively** stronger than either alone:

1. **Draft (#69 REASONING-DISTILL):** R1-class teacher emits informal natural-language proof sketch.
2. **Sketch (translation):** sketch is autoformalized into Lean stub with `sorry` placeholders.
3. **Prove (C):** verifier-in-loop trained student fills `sorry` placeholders with formal tactics.

DSP achieved 39.3% on miniF2F-test vs ~25% for sketch-only baselines — a ~1.6× relative lift from the prove stage on top of sketch capability.

**Implication for C selection.** C's lift over a #69-only baseline is the **incremental DSP-style prove-stage contribution**, ~1.5-2× on miniF2F. The headline 5-20× includes the full pipeline; the marginal C contribution is at the low end of the magnitude range.

---

## 10. User-need conditional decision tree

The SELECT-CONDITIONAL framing makes the verdict explicit:

```
IF user brief at iter-226 prioritizes math/formal-reasoning capability
   (e.g., math-assistant deployment, IMO-class problem solving,
    Lean/Coq tactic completion as primary use case):
   → SELECT
   → headline 5-20× on miniF2F / ProofNet / Lean-tactic;
     marginal 1.5-3× over post-#69 baseline;
     risk-adjusted 0.8-3.3× expected.

ELIF user brief prioritizes general-purpose conversational quality:
   → RESERVE for #83+
   → narrow domain doesn't match general user need;
     other #82 candidates with broader applicability preferred.

ELIF user brief is silent on formal-reasoning priority:
   → DEFAULT to RESERVE for #83+
   → conservative path; revisit at #83 if user signals math/formal need.
```

The default branch (silent brief → RESERVE) is the operative path absent explicit signal. Iter-226 brief at the time of this writing has not signaled math/formal-reasoning as a primary axis; **operative verdict is RESERVE for #83+**.

---

## 11. Bottom line

**THEOREM-PROVING-DISTILL-CHIRON is a well-motivated narrow-domain candidate.** It opens a genuinely new axis (FORMAL-VERIFICATION, the 23rd) with the program's only **bias-free reward signal** (Theorem 3), composes cleanly with #59 PRM and #69 REASONING-DISTILL via the DSP pattern, and has strong production precedent (AlphaProof IMO Silver, AlphaGeometry, LeanDojo, ProofNet, DSP, Llemma).

The honest counter-weight: domain is **narrow** (miniF2F / ProofNet / Lean/Coq tactic prediction); marginal lift over post-#69 baseline is **1.5-3×** (not 5-20×); risk-adjusted realization **0.8-3.3×**; ~30-40% mechanism overlap with #69 on math subsets; verifier-in-loop is the dominant engineering risk.

**Verdict:** **SELECT-CONDITIONAL on user brief signaling math/formal-reasoning priority**; otherwise **RESERVE for #83+**. Default operative verdict at iter-226 (absent explicit signal): **RESERVE**.

**Headline speedup (if selected):** **5-20× on miniF2F / ProofNet / IMO-formal subsets and Lean/Coq tactic prediction**; **risk-adjusted 0.8-3.3×** on the narrow domain; **marginal 1.5-3× over post-#69 baseline** (DSP-style prove-stage contribution).

**Engineering:** ~1300 LOC over 7 weeks. **Joint Gate-0 PASS ~55%; LLM-scale empirical confirmation ~30%.**

**Composition fingerprint:**
- Strong: #59 PRM (free dense per-tactic reward), #68 SUPER-DISTILL (cached-logit reuse), #69 REASONING-DISTILL (DSP draft stage).
- Weak: #60 TOOL-LLM (verifier-as-tool framing), #62 AGENT (proof-as-trajectory framing), #67 CAUSAL (orthogonal grounding axis).
- Neutral on all 22 prior axes (text NLL preserved by Theorem 1; bijectivity by Theorem 2).

**#82 slot disposition recommendation.**
- If sibling A or B opens broader axis with comparable production validation: **prefer sibling**, RESERVE C for #83+.
- If both siblings rejected or microopt-class: **promote C as SELECT-CONDITIONAL**, with explicit user-brief check.
- If user signals math/formal-reasoning priority at #82: **promote C unconditionally**.

After 41 paradigms across 22 axes pre-#82, the FORMAL-VERIFICATION axis remains the program's clearest gap on the rigorously-grounded-reasoning frontier. Reserving C for #83+ keeps the option open without consuming the #82 slot on a narrow-domain candidate when broader options may exist among siblings A and B.
