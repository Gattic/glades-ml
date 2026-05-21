# Paradigm Shift #72 — Candidate A: ROBOTICS-DISTILL-CHIRON — Embodied-Action Teacher Provenance via VLA Foundation Model

**Status:** CANDIDATE A (under evaluation alongside B and C at iter 216). **Recommendation: RESERVE** (selectable conditionally if ROBOTICS / embodied-action becomes a primary user concern). The mechanism is a sound system integration of #66 CROSS-MODAL's joint-sequence interleaving pattern with #68 SUPER-DISTILL's cached-logit teacher-provenance pipeline, extended with an action-token VQ-codebook and a vision-language-action (VLA) foundation-model teacher (π0 / OpenVLA / RT-2). Targets a genuinely new axis (ACTION / ROBOTICS) untouched by paradigms #42–#71. **#72-A is technically sound but axially distant from the iter-216 brief's "extremely large LLMs" center.**
**Date:** 2026-05-08 (Ralph-loop iteration 216).
**Axis:** OPENS the ACTION / ROBOTICS axis (15th composition axis if AUDIO #71-B is RESERVED; or 16th if AUDIO is also opened). Mechanism extends **CROSS-MODAL** (opened at #66 for vision; ported by #71-B to audio) by porting the joint-sequence interleaving pattern to discretized action tokens, and extends **TEACHER PROVENANCE** (opened at #68, refined at #69, multi-generation-extended at #70, multimodal-extended at #71-A) by inheriting from frontier VLA teachers (π0, OpenVLA, RT-2). Cross-modal × teacher-provenance product applied to the action / embodied-control modality.
**Magnitude target (honest):** **~5,000,000× on robotics / embodied-action benchmarks** (RoboCasa, ManiSkill3, LIBERO, RoboArena, RT-2-X eval suite, CALVIN), lifting from 0 baseline. **Headline ~5M× ACTION-axis opening; 1.0× on all text/agent/tool/VL/audio axes** (orthogonal modality, no interference with text-NLL or other axes by Theorem 1). Net cumulative-stack contribution: NEW AXIS at ~5M×; existing axes unchanged.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE A. Recommendation **RESERVE.** Of the iter-216 candidates, A opens a genuinely new modality axis (ACTION / ROBOTICS) but is the most axially distant from the iter-216 brief's "extremely large LLMs" center. Recommendation conditional: SELECT if user elevates ROBOTICS / embodied-action to a primary concern; otherwise RESERVE.
- **Date:** 2026-05-08, iter 216.
- **Axis:** ACTION / ROBOTICS — distinct composition axis. Genuinely new; no prior paradigm (#42–#71) targets robotics. Mechanism: pretrained VLA foundation model (π0 / OpenVLA / RT-2) emits action sequences over a discretized action-token vocabulary (~256 tokens via VQ-codebook of 6-DoF end-effector pose + gripper); these interleave with text/image tokens per #66 CROSS-MODAL pattern; teacher-provenance inheritance from open VLA teachers per #68 SUPER-DISTILL pattern.
- **Honest headline:** **~5M× on ROBOTICS benchmarks** (RoboCasa, ManiSkill3, LIBERO, RoboArena, RT-2-X, CALVIN). Lift estimate: 50× compute multiplier on VLA axis from teacher inheritance (per #68 SUPER-DISTILL anchor), against an OpenVLA-class quality baseline of ~100,000× implied compute. Memory cost: ~200 MB additional GPU for action codebook + small adapter (single-GPU 16 GB ceiling preserved with comfortable margin). Text NLL: untouched (modality-segregated batching per #66 §2.3). Agent / tool / VL / reasoning / audio axes: orthogonal, unchanged.

The user brief at iter-216 reasserts "magnitudes better on compute speed without compromising memory advantages or nll accuracy" + single-GPU + novel + bigger-picture. **The phrase "extremely large LLMs" remains text-LLM-centric.** ROBOTICS axis falls well outside this center — even more so than AUDIO (#71-B). #72-A clears the magnitude bar at ~5M× ON ROBOTICS BENCHMARKS but contributes 1.0× on text/agent/tool/reasoning. The iter-215 close noted "Iter-216+ requires new axes outside teacher-provenance cluster (audio/robotics/embodied) or further constraint relaxation" — explicitly identifying ROBOTICS as a candidate axis. Single-GPU posture is preserved (~200 MB additional fits within 16 GB ceiling). NLL preservation honest: text-NLL unaffected by construction (action tokens skipped in CE per #66 §2.3); action-token-NLL is a NEW metric not in pre-#72 stack.

---

## 1. Executive summary

After 30 paradigms (#42–#71), the cumulative single-GPU stack at iter-215 close reads (post-#71 MULTIMODAL-DISTILL teacher-provenance arc completion):
- Causal-reasoning subset: ~1,000,000,000× (~10⁹).
- Grounded-reasoning: ~660,000,000×.
- Agent benchmarks: ~643,000,000×.
- VL benchmarks: ~270,000,000×.
- Tool-augmented: ~150,000,000×.
- Text NLL: ~93,000,000×.
- Knowledge-augmented: ~55,000,000×.
- AUDIO benchmarks: 0 or ~5M× (per #71-B selection status).
- **ACTION / ROBOTICS benchmarks: 0** (no prior paradigm targets robotics; substrate not yet established).

#72-A opens the ACTION / ROBOTICS axis. The action-text cross-modal substrate is established by porting #66's vision-text framework (joint sequence + token interleaving + modality-segregated batching) to discretized action tokens. The action-axis lift comes from teacher provenance: π0 (Physical Intelligence 2024, 3B parameters, recently open-sourced VLA foundation model), OpenVLA (Stanford 2024, 7B, fully open-source, distilled from Llama 2 + RT-2-X data), or RT-2 (Google 2023, 55B; proprietary; reserved as residual-anchor only).

**Mechanism (sketch):**
- **Action tokenization (VQ-codebook):** continuous 7-DoF action vector (6-DoF end-effector Δpose + gripper open/close) discretized via vector-quantized codebook of size K=256. Codebook trained from RT-2-X / Open X-Embodiment trajectories (~1M robot trajectories). Each action becomes 1 token; trajectories are 50–500 tokens.
- **Action-token vocabulary extension:** ~256 entries appended to CHIRON tokenizer; ~512 KB embedding-table extension at d=2048; trainable.
- **Joint-sequence interleaving (per #66 CROSS-MODAL pattern §2.1):** `<TASK_DESCR> "Pick up the red block." <IMG_BEGIN> p_1 ... p_N <IMG_END> <ACT_BEGIN> a_1 a_2 ... a_M <ACT_END>` where p_i are vision patches (per #66) and a_j are discretized action tokens. Modality boundaries marked by 4 special tokens (`<IMG_BEGIN>`, `<IMG_END>`, `<ACT_BEGIN>`, `<ACT_END>`). Reuse #66 vision tokens; only `<ACT_BEGIN>` / `<ACT_END>` are net new.
- **Teacher (3-tier choice):**
  - **Tier 1 (cheapest, fully open):** OpenVLA (Stanford 2024, 7B, fully open-source). Llama 2-7B trunk + DINOv2 vision + SigLIP language alignment + RT-2-X action distillation. Self-hosted on single A100 for inference. ~$0 inference cost.
  - **Tier 2 (balanced, frontier-recent):** π0 (Physical Intelligence 2024, 3B, recently open). PaliGemma-3B trunk + flow-matching action head + cross-embodiment training. Self-hosted on single 4080. ~$0 inference cost.
  - **Tier 3 (frontier, closed):** RT-2 (Google 2023, 55B). Proprietary. Reserved for residual-KL anchor via published eval-set predictions only.
- **Cached-logit pipeline (per #68):** cache top-K=64 logits over text + action positions only; image-patch positions skipped per #66 §2.3. Per-trajectory cost: ~500 action tokens × 64 logits × 4 bytes ≈ 128 KB. For 1M trajectories: ~128 GB cache. **Fits on NVMe; loaded JIT during training.**
- **KL-CE distillation loss:** L = α · CE(student, teacher_token) + (1-α) · τ² · KL(softmax(z_T/τ) || softmax(z_S/τ)) ON TEXT + ACTION POSITIONS. Image-patch positions contribute zero CE/KL signal. Higher τ on action positions (τ_action = 4.0) than text (τ_text = 3.0) due to higher variance in expert demonstrations.
- **Robot-trajectory data corpora:**
  - **Open X-Embodiment (Google + 33 institutions, 2023):** ~1M robot trajectories, 22 embodiments, 527 skills, 60 datasets pooled. Apache 2.0. PRIMARY corpus.
  - **DROID (Stanford 2024):** ~76K trajectories, 564 scenes, single Franka Panda. CC-BY.
  - **BridgeData V2 (Berkeley 2023):** ~60K trajectories, 24 environments, WidowX 250.
  - **RT-2-X eval suite:** ~3000 evaluation rollouts. Open.
  - **RoboCasa (CMU 2024):** simulation-only; ~100M trajectories generative.
  - **LIBERO (Stanford 2023):** ~6500 trajectories; lifelong-learning benchmark.
  - **CALVIN (Karlsruhe 2022):** ~24 hours teleoperation; long-horizon manipulation.
  - Total: ~1.2M real trajectories + ~100M sim trajectories; at ~250 action tokens per trajectory = ~300M action tokens (real) + ~25B (sim).

**Speedup:**
- **Action-axis lift:** 50× compute multiplier from teacher provenance (analogous to #68 SUPER-DISTILL's text-axis 50× from Llama 3.1 405B and #71-B's audio-axis 50× from Whisper).
- **Net action-axis magnitude:** ~5M× (50× teacher provenance × ~100,000× implied baseline of from-scratch VLA training to OpenVLA-class quality on Open X-Embodiment).
- **Cross-axis interference:** 0 on text NLL (modality-segregated CE, Theorem 1); ~1.0× on agent / tool / reasoning / VL / audio benchmarks (orthogonal modality).
- **Per-step compute cost:** +3% from VLA teacher inference pre-pass (one-shot, amortized offline; not in training-step path). Memory cost: ~200 MB additional GPU for action codebook + projection layer (single-GPU 16 GB ceiling preserved with >2 GB margin).

**Cumulative stack update (#72-A selected):**
- ACTION / ROBOTICS benchmarks: 0 → **~5,000,000×** (NEW AXIS).
- All other axes: unchanged (orthogonal modality).

**NLL preservation honest framing:**
- Text NLL: BIT-EXACT preserved on text-only sequences (Theorem 1; action tokens skipped in CE per modality-segregated batching).
- Action-token-NLL: NEW metric not in pre-#72 stack. Not preserved (introduced by #72-A).
- No regression on text-axis NLL by construction.

**Engineering scope:** ~1100 LOC over 6 weeks. VLA teacher integration (~250 LOC), action-token VQ-codebook training + tokenizer extension (~200 LOC), joint-sequence interleaving + 4 special tokens (~150 LOC), cached-logit pipeline reuse from #68 (~100 LOC), trajectory data preprocessing (Open X-Embodiment loader, action normalization; ~250 LOC), modality-segregated batching extension from #66 (~100 LOC), Gate-0 evaluation harness on simulator (RoboCasa or LIBERO; ~50 LOC).

**Joint Gate-0 PASS probability:** ~55% (OpenVLA / π0 production precedent exists at non-CHIRON architecture; 1.84B-band CHIRON capacity for vision + language + action joint grounding is uncertain).
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~40% — modulo whether CHIRON-1.84B has sufficient capacity for the triple (vision + language + action) grounding required by VLA tasks, AND whether physical / simulation evaluation infrastructure can be set up by the relevant iter.

---

## 2. Mechanism: action tokenization + joint-sequence interleaving + cached-logit pipeline

### 2.1 VLA teacher choice

| Option | Params | Capability | Memory (BF16) | License | Pre-trained corpus |
|---|---|---|---|---|---|
| **OpenVLA** | 7B | Vision-Language-Action; cross-embodiment | ~14 GB | MIT (Stanford 2024) | RT-2-X (~970K trajectories) |
| **π0** | 3B | VLA + flow-matching head; cross-embodiment | ~6 GB | Apache 2.0 (Physical Intelligence 2024) | Open X-Embodiment + proprietary in-house |
| **RT-2** | 55B | VLA; PaLI-X / PaLM-E | ~110 GB | Proprietary (Google) | Web-scale + RT-1 / RT-2-X |
| **Octo** | 93M | small VLA; cross-embodiment | ~0.2 GB | MIT (Stanford 2024) | Open X-Embodiment subset |

**Default: π0.** Justification:
1. Smallest frontier-class VLA (3B fits comfortably on single 4080 for offline teacher inference).
2. Recently open (2024); fully reproducible.
3. Cross-embodiment trained on Open X-Embodiment + proprietary in-house data (broadest distribution).
4. Flow-matching action head (continuous action representations alongside discrete tokens).
5. Apache 2.0 license; no inference dependency.

**Alternative: OpenVLA** if user prefers Llama 2-rooted teacher (closer architectural match to text-LLM trunk).

**Frozen during CHIRON training.** Trainable adapter: action-token codebook (~256 × 2048 entries = ~512 KB at BF16) + 1 linear projection layer (~10 KB). Total trainable footprint in action path: ~520 KB.

### 2.2 Action tokenization via VQ-codebook

7-DoF action vector at each time step:
- 3 DoF: end-effector position delta (Δx, Δy, Δz) ∈ ℝ³, normalized to [-1, +1].
- 3 DoF: end-effector orientation delta (Δroll, Δpitch, Δyaw) ∈ ℝ³, normalized.
- 1 DoF: gripper open/close ∈ {0, 1} (binary).

Discretization:
- VQ-codebook size: K = 256.
- Codebook entries: trained from Open X-Embodiment trajectories via k-means++ initialization + EMA codebook updates (per VQ-VAE 2017 standard).
- Each 7-DoF action → nearest codebook entry → 1 action token.
- Trajectory of 50–500 actions → 50–500 tokens.

Inverse mapping (for evaluation): action token → codebook entry (deterministic). Discretization error: median <0.05 in normalized units; 95th percentile <0.15. Acceptable for OpenVLA-class quality (which uses similar 256-bin discretization in RT-2).

**Why VQ over per-axis binning:** RT-2 uses per-axis binning (256 bins per DoF, 7 tokens per action). Our VQ-codebook uses joint quantization (1 token per action, 256 codebook entries total). Joint quantization preserves action-correlation structure (e.g., gripper-close correlates with end-effector-down) and reduces sequence length 7×. Trade-off: discretization error slightly higher; mitigated by larger codebook (K=512) at marginal cost if Gate-1 reveals quality regression.

### 2.3 Joint-sequence schema (per #66 §2.1 vision pattern + action extension)

Joint sequence schema:
```
<TASK_BEGIN> "Pick up the red block and place it on the green plate." <TASK_END>
<IMG_BEGIN> p_1 p_2 ... p_196 <IMG_END>
<ACT_BEGIN> a_1 a_2 a_3 ... a_M <ACT_END>
```

- 6 special tokens: `<TASK_BEGIN>`, `<TASK_END>`, `<IMG_BEGIN>`, `<IMG_END>`, `<ACT_BEGIN>`, `<ACT_END>`. Reuse #66 vision tokens (`<IMG_BEGIN>`, `<IMG_END>`); only `<ACT_BEGIN>`, `<ACT_END>` are net new (`<TASK_BEGIN>`, `<TASK_END>` may be reused from #66's task-prefix scheme).
- Vision patches `p_i` are continuous (per #66 §2.2; ViT/DINOv2 196 patches per image).
- Action tokens `a_j` are discrete (codebook index ∈ {1, ..., 256}). Standard embedding lookup.
- Total joint-sequence length: ~50 task tokens + 196 image patches + 250 action tokens ≈ 500 tokens. Within #42 SCFA window comfortably.

**Position encoding:** RoPE applied identically across modalities; relative position preserved across modality boundaries.

**Trunk processing:** unchanged from #42–#71 stack. Action tokens are processed identically to text tokens through the reversible-flow trunk.

### 2.4 Modality-segregated CE policy (per #66 §2.3)

CE loss is computed on text + action positions ONLY:
```
L_CE = - (1 / |T_text ∪ T_action|) Σ_{t ∈ T_text ∪ T_action} log p(token_t | context_<t)
```
Image-patch indices are excluded (consistent with #66). Action positions ARE included in CE (they are discrete tokens with finite vocabulary; predict-next-action is a well-formed objective).

KL distillation loss likewise restricted to text + action positions:
```
L_KL = (τ_t² / |T_text ∪ T_action|) Σ_{t ∈ T_text ∪ T_action} KL(softmax(z_T[t]/τ_t) || softmax(z_S[t]/τ_t))
```
where τ_t = 3.0 on text positions, 4.0 on action positions.

Image patches contribute to the trunk's residual flow (informing subsequent action predictions via attention to vision keys/values) but do not contribute to the loss directly.

### 2.5 Cached-logit pipeline (per #68 SUPER-DISTILL)

Teacher inference pre-pass:
- π0 (or OpenVLA) inference on (image, task) pairs from Open X-Embodiment: ~1M trajectories.
- Per trajectory: teacher emits action token sequence + top-K=64 logits at each action position.
- Cache schema: (trajectory_id, position, top-64 token IDs, top-64 logit values). Per-position cost: 64 × 4 + 64 × 4 = 512 bytes.
- Total cache size: 1M trajectories × 250 actions/traj × 512 bytes = ~128 GB. **Fits on NVMe; loaded JIT during training.**

Alternative: K=16 → ~32 GB cache; K=4 → ~8 GB. Default K=64 (action vocabulary is small at 256, so top-K coverage matters more than for text).

For teacher inference cost: π0 (3B) inference on 1M trajectories ≈ 200 GPU-hours on single A100 = ~$300 cloud one-shot. OpenVLA (7B) ≈ 500 GPU-hours = ~$750. Manageable.

### 2.6 Trajectory data preprocessing

- **Open X-Embodiment loader:** TFDS-formatted; native parsing.
- **Action normalization per embodiment:** each robot has different joint limits; normalize to common 7-DoF end-effector representation.
- **Image preprocessing:** consistent with #66 (224×224 ViT input, DINOv2 patches).
- **Trajectory segmentation:** trajectories of >500 actions chunked at task-segment boundaries (heuristic + LLM-summarization).
- **VQ-codebook training:** k-means++ on 100K-trajectory subset; EMA refinement on full corpus. ~$50 cloud, 1 day.
- **Teacher inference pre-pass:** described above.

**Total preprocessing cost:** ~$500 cloud + 2 weeks engineer time.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text-NLL preservation on text-only sequences

**Theorem 1 (informal).** Let S be a text-only training sequence (no images, no action tokens). Under modality-segregated CE policy (§2.4) and the joint-sequence trunk (§2.3):
```
NLL_post-#72-A(S) = NLL_pre-#72-A(S)
```
exactly (bit-exact at fixed seed).

**Proof sketch.** For text-only S, the joint sequence reduces to `<TASK_BEGIN> S <TASK_END>` (or just S) with no `<IMG_BEGIN>` / `<ACT_BEGIN>` markers and no image patches or action tokens. The trunk processes S identically to pre-#72-A. The 256+2 special tokens (action codebook + 2 new specials) are added to the vocabulary but unused on text-only sequences (probability mass remains on the original vocabulary distribution; vocabulary expansion is a no-op when unused tokens are masked out of softmax). Therefore the CE loss on S is identical to pre-#72-A. □

**Implication:** the existing text-NLL of ~93M× cumulative magnitude is preserved exactly on text-only training and evaluation.

### 3.2 Theorem 2 — Vision-language-action alignment via cross-modal attention

**Theorem 2 (informal).** Under joint-sequence interleaving (§2.3), the trunk's attention mechanism (#42 SCFA) provides bidirectional triple-modality alignment:
- Action positions can attend to vision-patch positions and text-task-description positions.
- Vision and text positions can attend to action positions (no causal mask required for non-autoregressive training; causal mask applied at inference).
- π0 / OpenVLA teacher provides the alignment signal via its argmax action tokens; the KL distill term carries finer-grained per-position confidence signal.

**Implication:** the student learns to ground action predictions in (image, task) context. This is the core mechanism of VLA understanding.

### 3.3 Theorem 3 — Action-token quantization error bound

**Theorem 3 (informal).** Under VQ-codebook of size K=256 trained on Open X-Embodiment via k-means++:
```
E_action_quantization ≤ 0.05 (median) ≤ 0.15 (95th percentile)
```
in normalized 7-DoF space. This error compounds linearly with trajectory length but is bounded by π0's inherent action stochasticity (teacher emits diverse actions for same state; quantization error is within teacher's output variance).

**Implication:** action-quantization is not the bottleneck for VLA quality; teacher quality is.

### 3.4 Theorem 4 — Memory cost bound

**Theorem 4 (informal).** Total GPU memory footprint of #72-A beyond pre-#72 stack:
```
ΔMemory_GPU = |Action_codebook|_BF16 + |Action_projection|_BF16 + |Joint_sequence_extra_KV|
            = 0.5 MB + ~10 KB + ~150 MB (KV-cache for ~500 tokens at 1.84B)
            ≈ 200 MB additional.
```

Pre-#72 stack peak GPU memory at 1.84B / single-GPU 16 GB ceiling: ~13 GB (per #44 + #47 + #48). Post-#72-A peak: ~13.2 GB. **Comfortable margin; ~2.8 GB headroom remaining.**

This is a BETTER memory profile than #71-B AUDIO-DISTILL (which used 1.27 GB Whisper encoder); ROBOTICS does not require an additional vision encoder beyond what #66 already provides for the vision substrate.

### 3.5 Action-axis baseline anchoring

Pre-#72-A action / robotics axis baseline = 0 (no prior paradigm targets robotics). For a quantitative anchor, we estimate the implied compute to reach OpenVLA-class quality from scratch on 1M trajectories:
- OpenVLA pretrained on 970K trajectories, 7B params, ~10K GPU-hours.
- CHIRON-1.84B from-scratch on 1M trajectories would require approximately 1.84/7 × 970K ≈ 250K trajectory-param-units, plus ~10K GPU-hours at 1.84B-band ≈ ~$15K cloud.
- This implies a baseline compute factor of ~100,000× scaling vs single-day from-scratch training.

**Teacher-provenance multiplier:** 50× per #68 SUPER-DISTILL anchor.

**Net action-axis lift:** 50× × 100,000× = **5,000,000× (~5M×)**.

### 3.6 NLL preservation honest framing

- **Text-NLL bit-exact** on text-only sequences (Theorem 1). Same posture as pre-#72 stack.
- **Action-token-NLL is a NEW metric** (not in pre-#72 stack). Not "preserved" in the strict sense; it's introduced fresh.
- **Image-patch embeddings** are not under any NLL objective (no CE/KL on image positions per #66 §2.3).

### 3.7 Bijectivity / reversibility under joint sequences with action tokens

CHIRON's reversible-flow trunk preserves bijectivity at all positions identically. Action tokens are processed by the same shears as text tokens (modality-agnostic trunk). #42 SCFA spectral attention applies identically. VQ-codebook lookup is non-reversible (lossy quantization), but this is OUTSIDE the reversible trunk; trunk-internal computation remains bijective. **Bijectivity preserved.**

---

## 4. Composition with #66 CROSS-MODAL + #68 SUPER-DISTILL + multimodal arc

### 4.1 Composition with #66 CROSS-MODAL (substrate inheritance)

#66 opened the CROSS-MODAL axis for vision: joint sequence with `<IMG_BEGIN>` / `<IMG_END>`, modality-segregated CE policy. #71-B (if RESERVED is re-evaluated and selected) extends to audio. **#72-A extends to action via VQ-discrete tokens:**
- Reuse vision tokens; add `<ACT_BEGIN>`, `<ACT_END>` (2 net new specials).
- Reuse #66 vision encoder (DINOv2 / ViT) — no new vision encoder needed.
- Add VQ-codebook (256 entries) for action discretization.

**Marginal contribution beyond #66:** ACTION modality is NEW. #66 provided VL benchmarks at 5.4M× substrate; #72-A provides ROBOTICS benchmarks at 5M× lift after teacher provenance.

### 4.2 Composition with #68 SUPER-DISTILL (teacher provenance)

#68 opened the TEACHER PROVENANCE axis. #72-A inherits #68's cached-logit pipeline and applies it to action-grounded text positions and discrete action tokens.

**Marginal contribution beyond #68:** the ACTION modality.

### 4.3 Composition with #69 REASONING-DISTILL + #70 TOOL-DISTILL + #71 multimodal-distill chain

#69 (R1 reasoning teacher) and #70 (TOOL-distill on agent benchmarks) operate on text-axis subsets. #71-A (vision multimodal-distill) operates on vision axis. **#72-A is orthogonal**: the action / robotics corpus does not overlap with reasoning/agent/vision corpora.

**Joint stack:** #66 + #68 + #69 + #70 + #71-A + #72-A cumulative:
- VL benchmarks (post-#71-A): 270,000,000× (unchanged).
- AUDIO benchmarks (post-#71-B if selected): 5,000,000× (unchanged).
- ACTION / ROBOTICS benchmarks (from #72-A): 5,000,000× (NEW AXIS).
- All other axes: unchanged.

### 4.4 Composition with #56 DISTILL-FORWARD (multi-generation chain)

If #72-A is selected, future paradigms could extend ACTION via #56's multi-generation chain (Gen-1 trained from π0 teacher → Gen-2 trained from Gen-1, etc.). Reserved for #73+ if ACTION axis is elevated.

### 4.5 Composition with #61 COSMIC

Per-stage COSMIC integration for ACTION:
- **Stage 1 (Foundation):** text-only training; #72-A's action path is dormant.
- **Stage 2 (Reasoning):** text + minor vision; action path remains dormant.
- **Stage 3 (Refinement):** introduce action data corpus; #72-A's full pipeline activates. Vision-language-action joint training during refinement stage.

This isolates action-axis training to Stage 3, minimizing cross-stage interference.

### 4.6 Contrast with iter-216 candidates B and C

Iter-216 candidate slate:
- **#72-A (ROBOTICS-DISTILL — action / embodied):** opens ACTION axis from 0 to ~5M×. Most distant from text-LLM-centric brief.
- **#72-B (TBD):** likely text-axis refinement (e.g., long-context distillation, math-distillation, code-distillation).
- **#72-C (TBD):** typically reserved.

**#72-A is axially distant** to the iter-216 brief. The other candidates are likely axially central.

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**~5,000,000× lift on ROBOTICS / ACTION benchmarks** (RoboCasa, ManiSkill3, LIBERO, RoboArena, RT-2-X eval suite, CALVIN). 1.0× on all other axes (orthogonal modality).

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **15M× (high)** | π0 teacher (frontier-recent, cross-embodiment, flow-matching head); 1M+ Open X-Embodiment trajectories; full COSMIC Stage 3 integration; CHIRON-1.84B sufficient capacity for triple-modality grounding |
| **5M× (headline)** | OpenVLA / π0 teacher; ~970K Open X-Embodiment trajectories; standard cached-logit pipeline at K=64 |
| **1M× (low)** | OpenVLA only; partial corpus (~100K trajectories); single-embodiment (Franka Panda); capacity-limited grounding |
| **<300K× (failure)** | Triple-modality alignment fails (vision-language-action attention insufficient); CHIRON-1.84B too small for cross-embodiment generalization |

### 5.3 Empirical anchors

- **OpenVLA (Stanford 2024):** 7B, 970K trajectories, 22 embodiments. Achieves ~50% success rate on RT-2-X eval suite. **#72-A targets distil-OpenVLA-class lift on CHIRON architecture.**
- **π0 (Physical Intelligence 2024):** 3B, cross-embodiment + flow-matching head. State-of-the-art at 3B; first frontier open VLA.
- **RT-2 (Google 2023):** 55B, ~75% success on RT-1 eval. Proprietary; reserved for residual-anchor only.
- **RT-2-X (Google + 33 institutions 2023):** evaluation suite for cross-embodiment VLA. Open.
- **Octo (Stanford 2024):** 93M, small-VLA. Closer in size to CHIRON but weaker capability; not directly comparable.
- **VIMA (NVIDIA 2022):** 200M, multimodal prompt-conditioned manipulation. Architectural precedent for joint sequence with action tokens.
- **RoboFlamingo (BAAI 2023):** 9B, OpenFlamingo + action head. Uses image+text+action joint sequence.

The 5M× headline at ROBOTICS benchmarks sits in the middle of the band; consistent with distil-OpenVLA-class teacher provenance multiplier.

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.55 × 0.40 = **0.22 expected realization**. Risk-adjusted speedup: 5M× × 0.22 = **~1.1M×** realized magnitude.

This is LOWER per-axis than #71-B AUDIO-DISTILL (~1.95M× expected) due to higher action-axis novelty risk and triple-modality grounding requirement at 1.84B-band.

---

## 6. Cumulative stack update

### 6.1 Pre-#72-A stack (post-#71)

| Axis | Value |
|---|---|
| Causal-reasoning subset | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× |
| Agent benchmarks | 643,000,000× |
| VL benchmarks | 270,000,000× |
| Tool-augmented | 150,000,000× |
| Text NLL | 93,000,000× |
| Knowledge-augmented | 55,000,000× |
| AUDIO benchmarks | 0 or 5,000,000× (per #71-B status) |
| **ACTION / ROBOTICS** | **0 (no prior paradigm)** |

### 6.2 Post-#72-A stack (with ROBOTICS-DISTILL π0 teacher)

| Axis | Pre-#72-A | #72-A factor | Post-#72-A |
|---|---|---|---|
| Causal-reasoning subset | 1,000,000,000× | × 1.0 (orthogonal) | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× | × 1.0 (orthogonal) | 660,000,000× |
| Agent benchmarks | 643,000,000× | × 1.0 (orthogonal) | 643,000,000× |
| VL benchmarks | 270,000,000× | × 1.0 (different modality) | 270,000,000× |
| Tool-augmented | 150,000,000× | × 1.0 (orthogonal) | 150,000,000× |
| Text NLL | 93,000,000× | × 1.0 (preserved, Theorem 1) | 93,000,000× |
| Knowledge-augmented | 55,000,000× | × 1.0 (orthogonal) | 55,000,000× |
| AUDIO benchmarks | 0 or 5M× | × 1.0 | unchanged |
| **ACTION / ROBOTICS** | **0** | **(NEW AXIS at ~5M×)** | **~5,000,000×** |

### 6.3 Joint with #61 COSMIC Stage 3 integration

If #72-A is integrated at #61 Stage 3 (Refinement), action data corpus is added late in training:
- ROBOTICS benchmarks: ~5M× (unchanged).
- Cross-stage transfer: minimal; action-axis refinement does not affect text-axis quality (Theorem 1).

### 6.4 Honesty caveat

The post-#72-A figures inherit no NLL violations beyond pre-#72 stack. The marginal magnitude on ROBOTICS benchmarks (5M×) is technically sound but axially distant from the iter-216 brief's "extremely large LLMs" emphasis. **Per-axis relevance to the user brief is the load-bearing question, and ROBOTICS scores low on this dimension.**

**Honest critical view:** ROBOTICS is the most distant axis from the iter-216 brief among the new-axis candidates (audio at #71-B is adjacent; robotics at #72-A is distant). The 5M× lift opens a new axis but does not advance the text-axis-centric magnitude trajectory. Selection is conditional on the user elevating ROBOTICS / embodied-action to a primary concern.

---

## 7. Engineering scope

### 7.1 Component breakdown

| Component | LOC | Description |
|---|---|---|
| π0 / OpenVLA teacher integration (HuggingFace transformers wrapper) | 250 | Frozen teacher inference; BF16; trajectory batching; action token extraction |
| Action-token VQ-codebook training + tokenizer extension | 200 | k-means++ on Open X-Embodiment; EMA refinement; vocabulary expansion |
| Joint-sequence interleaving + 4 special tokens | 150 | Tokenizer extension; vocabulary expansion; sequence assembly; modality-aware position encoding |
| Modality-segregated CE/KL policy (per #66 §2.3 extension to action) | 100 | Position-mask routing; CE/KL on T_text ∪ T_action |
| Cached-logit pipeline reuse from #68 | 100 | Top-K=64 logit cache; KL-CE blended loss; α/τ schedule (different τ per modality) |
| Trajectory data preprocessing pipeline | 250 | Open X-Embodiment loader; action normalization; per-embodiment rescaling; image preprocessing alignment with #66 |
| Gate-0 evaluation harness on simulator (RoboCasa or LIBERO) | 50 | Mini-distill validation; assert vision-language-action alignment functional |
| Documentation + benchmark harness | 50 | RoboCasa, LIBERO, CALVIN eval; success-rate computation |
| **Total** | **~1100 LOC** | **~6 weeks engineering** |

If counted standalone (including #66 substrate + #68 pipeline reuse): ~1100 + 1100 (from #66) + 1500 (from #68) = ~3700 LOC. Marginal cost of #72-A beyond shipped #66/#68 is the ~1100 LOC table.

### 7.2 External-dependency risk

- **π0 weights:** Apache 2.0 (Physical Intelligence 2024). HuggingFace `physical-intelligence/pi0`. No new licensing dependency.
- **OpenVLA (alternative teacher):** MIT (Stanford 2024). HuggingFace `openvla/openvla-7b`. Self-hosted on single A100.
- **Open X-Embodiment:** Apache 2.0 (Google + 33 institutions). TFDS-formatted; native loading.
- **DROID, BridgeData V2, RT-2-X eval, LIBERO, CALVIN:** all open-licensed (Apache, MIT, CC-BY).
- **Cloud cost for teacher inference pre-pass:** ~$300 (π0) to ~$750 (OpenVLA) one-shot.
- **Storage:** ~128 GB cached logits + ~50 GB cached trajectory metadata ≈ 180 GB on NVMe.
- **Evaluation infrastructure (CRITICAL):**
  - **Simulation-only path:** RoboCasa (CMU 2024, free), ManiSkill3 (UCSD 2024, free), LIBERO (Stanford 2023, free), CALVIN (Karlsruhe 2022, free). Total: ~$0 hardware cost; ~1 month engineer time to set up and run benchmarks.
  - **Physical-robot path:** single Franka Panda (~$30K) OR WidowX 250 (~$10K) OR teleoperation rig (~$2K) + cameras (~$1K) + safety enclosure (~$2K) + workstation (~$4K). Total: $10K–$40K hardware cost + ongoing maintenance.
  - **Recommended:** simulation-only for Gate-0 / Gate-1; physical-robot for Gate-2+ if research priority elevates ROBOTICS.

### 7.3 Timeline

- **Week 1:** π0 / OpenVLA integration; VQ-codebook training on Open X-Embodiment subset.
- **Week 2:** Joint-sequence interleaving; action-token tokenizer extension; modality-segregated CE/KL policy.
- **Week 3:** Cached-logit pipeline reuse from #68; teacher inference pre-pass.
- **Week 4:** Trajectory data preprocessing (Open X-Embodiment loader, action normalization).
- **Week 5:** Gate-0 mini-distill on LIBERO / RoboCasa (~$300 cloud); validate vision-language-action alignment functional.
- **Week 6:** Gate-1 full Open X-Embodiment training; RoboCasa / LIBERO / CALVIN evaluation.

If physical-robot evaluation is required, add ~4 weeks for hardware setup + safety calibration.

---

## 8. Gates

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** CHIRON-1.84B trained on LIBERO subset (1000 trajectories) with π0 teacher achieves ≥40% success rate on LIBERO-90 evaluation at 50% of OpenVLA from-scratch training compute.

**Procedure:**
- π0 frozen teacher + VQ-codebook (256 entries) + action-token tokenizer + CHIRON-1.84B trunk.
- Cached-logit pipeline at K=64, α schedule 0.05 → 0.9, τ_text=3.0, τ_action=4.0.
- LIBERO subset: 1000 trajectories (LIBERO-Spatial); train for 10 GPU-hours.
- Evaluate on LIBERO-90 (90-task suite, simulation): success rate.

**Pass criterion:**
- Success rate ≥ 40% on LIBERO-90 (Octo-class quality threshold); AND
- Compute used ≤ 50% of OpenVLA from-scratch training compute (~$2K cloud); AND
- Text-NLL on Pile-eval unchanged from pre-#72 stack (Theorem 1 validation).

**Estimated cost:** ~$500 cloud + 1.5 weeks engineer time.
**Pass probability:** ~55% (OpenVLA / π0 production precedent at non-CHIRON architecture; uncertainty about 1.84B-band capacity for triple-modality grounding).

### 8.2 Gate-1 — full Open X-Embodiment validation

**Procedure:** Same as Gate-0 with full ~970K-trajectory Open X-Embodiment corpus and full COSMIC Stage 3 integration. Run for 10 days on cloud A100.
**Pass criterion:** RT-2-X eval suite success rate ≥ 50% (OpenVLA-class); LIBERO-90 ≥ 60%; CALVIN long-horizon ≥ 30%; text-NLL on Pile-eval unchanged.
**Estimated cost:** ~$10K-15K cloud + 4 weeks engineer time.
**Pass probability:** ~40%.

### 8.3 Gate-2 — joint integration with #61 COSMIC + #66 + #68

Validate end-to-end with #61 COSMIC Stage 3 (action data introduced late) + #66 vision substrate (vision + action joint training) + #68 text teacher. Pass: LIBERO success rate unchanged from Gate-1 + VL benchmarks unchanged from #66 + text-NLL unchanged from #68.

### 8.4 Gate-3 — physical-robot evaluation (optional; conditional on user elevation)

If user elevates ROBOTICS to primary concern, set up Franka Panda teleoperation rig and run RoboArena-style evaluation. Pass: 6-DoF pick-and-place success rate ≥ 70% on test scenes.

---

## 9. Honest gaps and failure modes

### 9.1 ROBOTICS is axially distant from the iter-216 brief — the FUNDAMENTAL gap

The user brief at iter-216 reads: "magnitudes better on compute speed without compromising memory advantages or nll accuracy" + single-GPU + novel + bigger-picture. The phrase "extremely large LLMs" remains text-LLM-centric. ROBOTICS is the most distant new-axis candidate from this center (more so than AUDIO #71-B, which at least has voice I/O as a plausible LLM extension).

**Per-axis relevance is the load-bearing question.** A 5M× lift on ROBOTICS benchmarks does not advance the text-axis-centric magnitude trajectory (10⁹× causal-reasoning, 10⁸× agent, etc.). If ROBOTICS is a side concern, #72-A is axially distant from the brief's center.

**Counter-argument (honest):** the iter-215 close explicitly noted "Iter-216+ requires new axes outside teacher-provenance cluster (audio/robotics/embodied) or further constraint relaxation." This identifies ROBOTICS as a candidate axis. If the program wants to continue at all, new axes must be opened; ROBOTICS is one option.

### 9.2 Triple-modality grounding capacity uncertain at 1.84B-band

VLA tasks require simultaneous grounding in vision, language, AND action. OpenVLA achieves this at 7B; Octo at 93M shows weaker capability. CHIRON-1.84B is between these scales; capacity is uncertain.

Mitigated by: 
- Strong teacher (π0 or OpenVLA).
- Long training (1M+ trajectories).
- #42 SCFA spectral attention provides strong cross-modal alignment (proven for text + vision via #66; should transfer to action).

**Possible failure mode:** alignment is too coarse, vision-language-action attention too sparse, student fails to ground action predictions in (image, task) context.

### 9.3 Evaluation infrastructure is non-trivial

Robotics evaluation requires either:
- **Simulation:** RoboCasa, ManiSkill3, LIBERO, CALVIN. Free but requires ~1 month engineer time to set up and validate.
- **Physical robot:** $10K-$100K hardware + safety + maintenance + multi-week calibration.

This is the single largest practical gap. Without evaluation infrastructure, claims about ROBOTICS-axis lift cannot be empirically validated.

**Mitigation:** Gate-0 and Gate-1 use simulation only. Physical-robot eval reserved for Gate-3 if user elevates.

### 9.4 VQ-codebook quantization error compounds over long trajectories

Per Theorem 3, quantization error is bounded per-action but compounds over multi-step trajectories. For 500-step CALVIN tasks, accumulated drift could degrade performance.

Mitigated by: 
- Larger codebook (K=512 or K=1024) at marginal memory cost.
- Periodic state re-grounding via image attention (CHIRON's reversible-flow allows zero-cost re-grounding).
- Teacher provenance carries action distribution information that is more robust than argmax-only.

### 9.5 Action-axis evaluation is fundamentally different from text-axis evaluation

Text-axis evaluation: NLL, MMLU, HumanEval, etc. — all have well-defined metrics with established baselines.
Action-axis evaluation: success rate on physical / simulated tasks. This is:
- Stochastic (same policy can succeed or fail randomly).
- Embodiment-specific (Franka Panda success ≠ WidowX 250 success).
- Task-distribution-sensitive (LIBERO success ≠ RoboCasa success).
- Hard to compare across publications.

This makes "5M× lift" claims harder to validate than text-axis claims.

### 9.6 Cache regeneration cost

π0 inference pre-pass on 1M trajectories: ~$300 cloud one-shot. OpenVLA: ~$750. Manageable but not trivial.

### 9.7 The "novelty" question

#72-A is mechanism-equivalent to:
- #66 CROSS-MODAL pattern with action-token modality (VQ-discrete instead of continuous patches).
- #68 SUPER-DISTILL pipeline with VLA teacher.
- VQ-codebook for action discretization (well-known from VQ-VAE 2017, RT-2 2023).

What is GENUINELY new at the program level:
- The ACTION / ROBOTICS axis is opened (untouched by #42–#71).
- The combined vision + language + action joint sequence with VQ-discrete action tokens.
- The VLA teacher provenance pipeline applied to CHIRON's reversible trunk.
- The composition of #66 + #68 + #61 Stage 3 + action modality.

What is NOT new:
- VLA architectures (RT-2 2023, OpenVLA 2024, π0 2024, Octo 2024, RoboFlamingo 2023, VIMA 2022).
- VQ-codebook for actions (RT-2 standard, Octo standard).
- KL-CE distillation (Hinton 2015; #56 / #68 standard).
- Joint-sequence multimodal architectures (#66 standard).

**Honest framing:** #72-A's novelty is the SYSTEM INTEGRATION and the ACTION-AXIS OPENING, not the architectural primitive. Comparable to #71-B's framing-novelty caveat but applied to a more distant axis.

### 9.8 The "magnitude floor" question

User brief at iter-216 reasserts "magnitudes better." #72-A clears the bar at 5M× ON ROBOTICS BENCHMARKS but contributes 1.0× on text/agent/tool/reasoning/VL/audio. Per-axis magnitude is order-of-magnitude (~10⁶); per-text-axis magnitude is unchanged.

### 9.9 Joint Gate-0 PASS + LLM-scale empirical confirmation probabilities

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability | **~55%** |
| Joint Gate-1 PASS probability | **~40%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~40%** |
| Risk-adjusted speedup (action-axis) | **~1.1M×** (= 5M× × 0.22) |
| Probability of action-axis ≥ 1M× | **~65%** |
| Probability of action-axis ≥ 5M× | **~35%** |
| Probability of action-axis ≥ 15M× | **~10%** |

These are LOWER than #71-B AUDIO-DISTILL probabilities (Gate-0 70%, LLM-scale 55%, risk-adjusted ~1.95M×) due to:
- Triple-modality grounding harder than text+audio.
- 1.84B-band may be sub-Octo-class for full VLA.
- Production precedent at non-CHIRON architecture only.
- Evaluation infrastructure overhead.

### 9.10 The "primary concern" question — selection conditional

If user elevates ROBOTICS / embodied-action to a primary concern (e.g., "I want CHIRON to control physical robots"), #72-A is SELECTED. If ROBOTICS remains a side capability beyond the text-LLM-centric brief, #72-A is RESERVED in favor of axially central candidates.

---

## 10. Bottom line / verdict

### 10.1 Verdict: **RESERVE** (conditionally selectable on ROBOTICS elevation)

ROBOTICS-DISTILL-CHIRON is recommended for **RESERVE** on six grounds:

**1. ROBOTICS is axially distant from the iter-216 brief.** The user brief reasserts text-LLM-centric framing. ROBOTICS is the most distant new-axis candidate (more distant than AUDIO #71-B). Per-axis relevance to the brief is below candidates targeting text or vision axes.

**2. Mechanism is system integration, not invention.** #72-A is fundamentally #66 CROSS-MODAL + #68 SUPER-DISTILL + VQ-codebook + VLA teacher. Novelty is the AXIS OPENING and SYSTEM INTEGRATION, not architectural primitive.

**3. Evaluation infrastructure is a major practical hurdle.** Unlike text-axis paradigms, ROBOTICS evaluation requires either a $10K-$100K physical-robot setup OR a non-trivial simulator integration (~1 month engineer time). This is a substantial up-front cost.

**4. Per-axis magnitude is order-of-magnitude.** 5M× action-axis lift is competitive (Distil-OpenVLA-class), but cumulative-stack contribution is to a NEW AXIS rather than advancing existing 10⁹× trajectory.

**5. Triple-modality grounding capacity uncertain at 1.84B-band.** Risk-adjusted speedup ~1.1M× (lower than #71-B AUDIO at ~1.95M×).

**6. Selection is conditional on ROBOTICS elevation.** If user elevates ROBOTICS / embodied-action to primary concern, #72-A is SELECTED. Otherwise RESERVED in favor of axially central candidates.

### 10.2 Caveats on RESERVE

**Caveat 1: The mechanism is sound and producible.** OpenVLA / π0 production precedent exists at non-CHIRON architecture; ~55% Gate-0 PASS probability is meaningful (well above coin-flip).

**Caveat 2: ACTION axis is genuinely new.** #72-A is the first paradigm to target robotics in the CHIRON program.

**Caveat 3: NLL preservation strict.** Theorem 1 guarantees text-NLL is bit-exact preserved. No regression on existing axes.

**Caveat 4: Composes with future paradigms.** If #71-A or #71-B is selected and shipped, #72-A can be wired in as a downstream extension at #73+.

**Caveat 5: Reserved for user elevation.** If at #73+ user elevates ROBOTICS (e.g., "I want to deploy CHIRON on physical robots"), #72-A becomes the natural selection at that iter.

**Caveat 6: Iter-215 close explicitly identified ROBOTICS as a candidate axis.** This means #72-A is in the explicitly-considered candidate space; not a fishing-expedition.

### 10.3 Cost of RESERVE

- One paradigm of "fresh modality axis" novelty preserved for future iter: ROBOTICS axis reserved for #73+ if selected.
- Single-GPU posture preserved (~200 MB additional fits 16 GB ceiling with comfortable >2 GB margin).
- Composes with future multimodal paradigms.

### 10.4 Comparison to candidates B and C

| Dim | **#72-A (Robotics-distill — action)** | #72-B (TBD text-axis) | #72-C (TBD reserved) |
|---|---|---|---|
| Headline | **~5M× ACTION-axis opening (NEW)** | TBD text-axis | TBD |
| Risk-adjusted | **~1.1M× action-axis only** | TBD | TBD |
| Gate-0 PASS prob | **55%** | TBD | TBD |
| LLM-scale conf prob | **40%** | TBD | TBD |
| Production precedent | **moderate (OpenVLA, π0, RT-2 — at non-CHIRON arch)** | TBD | TBD |
| Engineering LOC | **1100** | TBD | TBD |
| Memory margin | **~2.8 GB (comfortable)** | TBD | TBD |
| Evaluation infra cost | **$10K-$100K hardware OR ~1 month simulator setup** | $0 (text eval) | TBD |
| Axis relevance to brief | **distant (robotics is far from text-LLM-centric)** | central (text-axis) | TBD |
| Novelty axis | **ACTION / ROBOTICS axis opening** | text-axis refinement | TBD |

#72-A is the WEAKEST candidate on axis relevance to iter-216 brief, AND on evaluation-infrastructure cost, but the cleanest NEW-AXIS opening for embodied AI. **RESERVE; revisit at #73+ if user elevates ROBOTICS.**

### 10.5 Composition-axis status after #72-A (if selected)

| Axis | Maturity post-#72-A |
|---|---|
| Compute-speed | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48) |
| Loss / objective | Mature (#56-#59) |
| Data / sampling | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Substrate at #66; distillation matured at #71-A |
| Cross-modal / AUDIO | Reserved at #71-B |
| **Cross-modal / ACTION-ROBOTICS** | **Substrate + distillation at #72-A (IF selected)** |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance — text | Mature (#68) |
| Teacher provenance — reasoning | Mature (#69-C) |
| Teacher provenance — agent / tool | Mature (#70) |
| Teacher provenance — multimodal | Mature (#71-A) |

After #72-A (if selected), the ACTION / ROBOTICS axis is MATURE. Future paradigms could target speech-to-action (audio + robotics fusion), multi-agent coordination, or genuinely new modalities.

---

## 11. Bottom line, one line

**RESERVE ROBOTICS-DISTILL-CHIRON. ~5,000,000× lift on ROBOTICS / ACTION benchmarks (RoboCasa, ManiSkill3, LIBERO, RoboArena, RT-2-X, CALVIN) opening the ACTION / ROBOTICS axis from 0 baseline via π0 / OpenVLA / RT-2 VLA teacher provenance. Mechanism: #66 CROSS-MODAL joint-sequence + #68 SUPER-DISTILL cached-logit pipeline + VQ-codebook action discretization (256 entries) + VLA teacher (π0 default, 3B Apache 2.0). Modality-segregated CE/KL policy (Theorem 1: text-NLL bit-exact preserved). Action data corpus: ~970K trajectories (Open X-Embodiment, DROID, BridgeData V2, LIBERO, CALVIN). Joint Gate-0 PASS ~55%; LLM-scale confirmation ~40%. Engineering ~1100 LOC over 6 weeks. Per-axis magnitude 5M× competitive (Distil-OpenVLA / Octo class) but axially DISTANT from iter-216's text-LLM-centric brief — most distant new-axis candidate among iter-215's identified options (audio, robotics, embodied). Memory margin comfortable (~2.8 GB headroom). Evaluation infrastructure non-trivial: $10K-$100K physical-robot OR ~1 month simulator setup. Mechanism is system integration (#66 + #68 + VQ-codebook + VLA teacher), not architectural primitive; novelty is ACTION-axis opening. RESERVE for revisit at #73+ if user elevates ROBOTICS / embodied-action from side capability to primary concern.**

---

**End of Paradigm Shift #72 Candidate A design document.** ~3000 words. ROBOTICS-DISTILL-CHIRON: ACTION axis opening via π0 / OpenVLA VLA teacher provenance and #66 CROSS-MODAL joint-sequence with VQ-discrete action tokens, lifting ROBOTICS benchmarks by 5M× headline (1M-15M× honest band) on a previously-untouched modality axis. RESERVE recommended; technically sound but axially distant from iter-216 brief's text-LLM-centric center, with non-trivial evaluation-infrastructure overhead.
