# VESTA Literature Audit — Linear-Recurrence / SSM / Efficient-Attention LLM Alternatives

**Date:** 2026-05-19
**Purpose:** Mandatory baseline audit for the next-generation research program in `glades-ml`. Identifies the strongest existing baseline for each reference axis (R1–R8 from `newmodel.txt`) and the documented weaknesses where a novel mechanism could earn novelty. Cites primary papers; numbers are taken from the original papers' reported figures.

**Context:** Replaces ad-hoc baseline assumptions in EALRMN (`research/EALRMN_WRITEUP.md`). EALRMN's Phase-1 ablation (`research/EALRMN_PHASE1_GPU_RESULTS.md`) confirmed that ~100% of its "wins" reduced to *linear-vs-tanh recurrence + orthogonal init* — a result already in this literature when EALRMN was specified.

---

## 1. Per-method summary

Each row: (inefficiency targeted) | (concrete advantage with regime) | (documented failure modes) | (C++ ref impl).

### S4 — Gu, Goel, Ré (NeurIPS 2021, arXiv 2111.00396)
1. **Targets:** quadratic attention; vanishing-gradient RNN at very long T. Uses HiPPO-initialized diagonal-plus-low-rank SSM with FFT-based long convolution.
2. **Advantage:** **~60× generation speedup** vs Transformer; **86.1% avg on Long Range Arena (LRA)**; first model to solve **Path-X (T=16k) at 96.4%** where all Transformers got chance accuracy. O(L log L) train, O(1) per-step inference.
3. **Failure modes:** time-invariant transition (LTI) → cannot do **selective copying** or **induction-head**; weak on associative recall (Zoology, Eyuboglu et al. 2023); fragile to LR schedule (Zoology).
4. **C++ impl:** None in glades-ml (only legacy `TYPE_RNN/GRU/LSTM` in `network.h`). Upstream is JAX/PyTorch (HazyResearch/s4). No widely used C++ port.

### S5 — Smith, Warrington, Linderman (ICLR 2023, arXiv 2208.04933)
1. **Targets:** S4's per-channel SISO bank + FFT-conv complexity. Replaces with a **single MIMO diagonal SSM** + parallel scan.
2. **Advantage:** **87.4% avg LRA**, **98.5% Path-X**; computational parity with S4 but cleaner formulation.
3. **Failure modes:** inherits all S4 LTI weaknesses (no selective recall).
4. **C++ impl:** None (JAX upstream, lindermanlab/S5).

### Mamba (S6) — Gu, Dao (COLM 2024, arXiv 2312.00752)
1. **Targets:** S4/S5 LTI limitation. Makes A, B, C **input-dependent (selective)** while keeping near-linear cost via hardware-aware selective scan.
2. **Advantage:** **5× higher throughput** than same-size Transformer at long T; **Mamba-3B matches Transformer-6B** on Pile pretraining and downstream evals. Solves the synthetic Selective Copy and Induction-Head tasks that LTI SSMs fail.
3. **Failure modes:** **multi-query associative recall (MQAR)** — Based (Arora et al. 2024) beats Mamba by **+32.2 acc on MQAR** and **+10.36 acc on real recall-heavy slices of Pile**. Also documented weak on exact copy / chain-of-thought (Jelassi et al. 2024, arXiv 2410.03810). Fixed-size recurrent state → "memory cliff" at moderate distractor counts.
4. **C++ impl:** None in glades-ml. Upstream is Python+Triton (`state-spaces/mamba`, custom `selective_scan_cuda`). No mature C++ port.

### Mamba-2 (SSD) — Dao, Gu (ICML 2024, arXiv 2405.21060)
1. **Targets:** Mamba's selective-scan being matmul-poor and hard to scale state dim. Establishes **Structured State-Space Duality (SSD)**: a restricted class of SSMs ≡ a restricted linear attention.
2. **Advantage:** **2–8× faster** than Mamba-1's selective scan; **6× faster than FlashAttention-2 at T=16k** (crossover at T=2k); allows state dim N=64/128 vs Mamba-1's N=16. Iso-quality with Mamba-1 on language modeling.
3. **Failure modes:** still SSM-class → same associative-recall weaknesses as Mamba; SSD imposes a restriction (scalar-times-identity decay per head) that loses some Mamba-1 expressivity.
4. **C++ impl:** None. Python + custom CUDA upstream.

### RWKV — Peng et al. (EMNLP 2023, arXiv 2305.13048) + Eagle/Finch (RWKV-5/6, arXiv 2404.05892) + Goose (RWKV-7, 2025)
1. **Targets:** softmax-attention's O(L²) inference cost. Reformulates linear attention as a **time-mixing recurrence** with **O(L) train, O(1) infer**.
2. **Advantage:** RWKV-4 14B was the first attention-free model scaled to Transformer-class size with competitive perplexity on Pile. RWKV-6 (Finch) introduces matrix-valued state + dynamic recurrence; RWKV-7 (Goose) claims SoTA on multilingual at 3B.
3. **Failure modes:** Same family as Mamba on associative recall (RWKV-4 lags Transformers on Pile recall slice per Zoology). Long-context retrieval improvements come from architecture iteration (v5→v7) not free.
4. **C++ impl:** **Best-in-class for our purposes.** `rwkv.cpp` (saharNooby/rwkv.cpp) ports RWKV-4/5 inference to GGML in pure C/C++. Training is still Python. glades-ml has no RWKV code.

### LRU — Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (ICML 2023, arXiv 2303.06349, "Resurrecting RNNs")
1. **Targets:** the question "is the HiPPO machinery in S4 actually necessary?" Strips SSM down to a **diagonal complex-valued linear RNN** with careful parameterization + normalization.
2. **Advantage:** **matches S4/S5 on every LRA task** with a plain linear RNN. This is the most important paper for the VESTA program — it isolates the variables.
3. **Failure modes:** still LTI; inherits SSM recall weaknesses; no selectivity.
4. **C++ impl:** None. JAX upstream (NicolasZucchet/minimal-LRU). **Cheapest C++ port** of any SSM-class model (~200 LOC), because the math is just diagonal complex-A linear recurrence.

### Hyena — Poli et al. (ICML 2023, arXiv 2302.10866)
1. **Targets:** attention's O(L²). Replaces with **implicit long convolutions + multiplicative data-controlled gating**, O(L log L).
2. **Advantage:** **2× over FlashAttention-2 at T=8k, 100× over FlashAttention at T=64k**; ~**20% less training compute** to reach Transformer-quality at T=2k on WikiText103/Pile.
3. **Failure modes:** worst-in-class on associative recall in Zoology (gated-conv arch); FFT-conv has poor wall-clock at short T.
4. **C++ impl:** None.

### RetNet — Sun et al. (arXiv 2307.08621, Jul 2023)
1. **Targets:** "training-parallel ↔ inference-recurrent" tradeoff. Multi-scale **retention** mechanism removes softmax, supports 3 modes (parallel, recurrent, chunkwise).
2. **Advantage:** Paper claims **8.4× faster inference, 70% memory savings** vs same-size Transformer at 7B; competitive perplexity on Pile at 1.3B–6.7B scale.
3. **Failure modes:** Reproduction reports (community) suggest the original training-parallel/recurrent equivalence is sensitive; recall weaknesses similar to other linear-attention variants. Less adopted than Mamba/RWKV in 2024–2026.
4. **C++ impl:** None upstream nor in glades-ml.

### Griffin / Hawk — De et al. (DeepMind, ICML 2024, arXiv 2402.19427)
1. **Targets:** pure-recurrence models' recall weakness + pure-attention's O(L²). Hybrid: **RG-LRU (real-gated linear recurrent unit)** + **local sliding-window attention** in alternating blocks.
2. **Advantage:** **Hawk-3B beats Mamba-3B** on downstream evals; **Griffin-14B matches Llama-2** with **6× fewer training tokens**; extrapolates to far longer sequences than train; lower inference latency + higher throughput than Transformer at scale.
3. **Failure modes:** still has the SSM-class recall ceiling outside the local-attention window; hybrid means it inherits some attention KV-cache cost (bounded by window).
4. **C++ impl:** None upstream nor in glades-ml.

### Mega — Ma, Zhou, Kong, Cui, Gu, May, Zettlemoyer, Ghorbani (ICLR 2023, arXiv 2209.10655)
1. **Targets:** softmax-attention's lack of position-aware locality bias. Adds a **multi-dim damped EMA** ahead of single-head gated attention. Mega-chunk variant gives linear cost.
2. **Advantage:** **88.21% avg LRA** (vs Transformer 59.24%, S4-v2 85.86%) — the strongest LRA number of the pre-Mamba era. Competitive on WMT, autoregressive LM.
3. **Failure modes:** EMA is a low-order linear filter — limited expressivity vs selective SSM; superseded operationally by Mamba.
4. **C++ impl:** None.

### Newer (2024–2026) work worth flagging
- **MambaByte** (Wang et al., COLM 2024, arXiv 2401.13660): **byte-level Mamba**. Matches subword Transformers on quality at byte granularity; with speculative decoding via subword drafter, recovers most of the tokenization throughput cost. This is the **strongest tokenization-free baseline** for R1.
- **Based** (Arora et al., ICML 2024, arXiv 2402.18668): **linear attention (Taylor approx of softmax) + sliding window**. **+10.36 acc** on real recall slices vs Mamba; **+32.2 acc on MQAR**. **24× throughput vs FA2 generation**. Strongest baseline for "recall-aware linear attention".
- **DeltaNet / Gated DeltaNet** (Yang, Schlag et al. 2024, arXiv 2406.06484): linear-transformer with a **delta-rule** update on a non-diagonal state. Yang et al. give a parallel-over-sequence training algorithm. Closes most of the recall gap vs softmax attention on synthetic MQAR.
- **Mixture-of-Depths** (Raposo et al., arXiv 2404.02258): **strongest compute-adaptive-inference baseline** for R7 — top-k token routing per layer. Iso-FLOPs lower-loss than dense.
- **Mixture-of-Recursions** (Bae et al., NeurIPS 2025, arXiv 2507.10524): per-token recursion depth + KV-cache sharing; ~**2.18× inference throughput** at iso-quality.
- **Mixtral 8×7B** + **Switch Transformer**: strongest sparse-MoE baselines for R5 (Mixtral 47B-total / 13B-active matches Llama-2-70B).

---

## 2. Axis × strongest-baseline table (R1–R8)

| Axis | Strongest existing-literature baseline | One-line justification |
|------|---------------------------------------|------------------------|
| **R1** entropy-adaptive chunking | **MambaByte** (arXiv 2401.13660) | Token-free byte-level SSM; the only widely-cited baseline that eliminates the surface-entropy/tokenization confound rather than learning chunk boundaries. Entropy-adaptive *patching* (e.g., Pagnoni et al. byte-latent transformer 2024) is a closer match conceptually but less load-bearing; MambaByte is the harder bar. |
| **R2** latent prediction | **No clean baseline exists** for autoregressive *language modeling* on a learned latent. JEPA / I-JEPA (LeCun 2022, Assran et al. 2023) is the cited reference but is non-generative and vision-centric. For LM, the field has not produced a published win for predicting `z_{t+h}` over next-token CE at iso-encoder-params. EALRMN Phase-0a falsified this; flag the absence. |
| **R3** novel recurrence | **Mamba-2 (SSD)** or **Griffin/Hawk** (depending on task). Mamba-2 is the strongest pure-recurrence baseline; Hawk wins on downstream evals at 3B and is the strongest *hybrid* baseline. LRU (Orvieto et al. 2023) is the strongest *minimal* baseline — any "novel recurrence" must beat LRU before claiming anything beyond it. |
| **R4** bounded associative memory | **Based** (Arora et al. 2024) for explicit recall; **Mamba-2** for "memory is just the recurrent state". Pure-SSM memory is *implicit in the state*; the closest *explicit* bounded-memory baseline in the linear-RNN line is Based's sliding-window + Taylor-linear-attention combo, which directly attacks MQAR. |
| **R5** sparse / mixture-of-experts | **Mixtral 8×7B top-k=2** (Jiang et al. 2024). Switch Transformer is the simpler reference. Any novel routing must beat top-k at iso-active-params on multi-distribution data. |
| **R6** memory-write regularization | **No clean baseline exists.** The closest published work is Based's compression-aware initialization and DeltaNet's delta-rule (which is *update* regularization, not *write* regularization). The axis is under-explored in published linear-RNN work — **opportunity for genuine novelty**. |
| **R7** compute-adaptive inference | **Mixture-of-Depths** (Raposo et al. 2024) and **Mixture-of-Recursions** (Bae et al. 2025). MoD is the cleanest iso-FLOPs comparison; MoR adds KV-cache sharing. Early-exit transformers (DeeBERT, etc.) are an older weaker baseline. |
| **R8** optional raw decoding | **No clean baseline exists.** Speculative decoding (Leviathan et al. 2023) and MambaByte's subword-drafter-byte-verifier come closest, but "decode latents → optionally decode tokens" is not a standard published axis. **Opportunity for genuine novelty** but requires careful task design or it collapses to speculative decoding. |

**Three axes (R2, R6, R8) have no strong literature baseline.** That is itself the most actionable finding of this audit.

---

## 3. Open problems where these methods are documented weak

The SSM / linear-RNN / linear-attention line has documented, *quantitative* failure modes. A novel mechanism that attacks one of these has a real chance of earning novelty.

1. **Multi-query associative recall (MQAR).** The cleanest documented weakness. Zoology (Eyuboglu, Arora, Zhang, Ré 2023; HazyResearch blog) showed sub-quadratic models lose to Transformers on real recall slices of Pile. Based (Arora et al., ICML 2024, arXiv 2402.18668) quantifies: **+32.2 acc on MQAR, +10.36 acc on Pile recall slices** vs Mamba. Mechanism: pure SSMs compress the past into a fixed-size state ("memory cliff"). **A novel R4 mechanism must clear or close this gap to count.**

2. **Exact copying / induction at very long T.** Jelassi, Brandfonbrener, Kakade, Malach (arXiv 2402.01032, "Repeat After Me" / 2024) and follow-up (Jelassi et al. arXiv 2410.03810) document Mamba's degradation on exact-copy and chain-of-thought tasks as T grows. Mamba "learns" induction heads on the synthetic Gu/Dao set but generalizes worse out-of-distribution than Transformer.

3. **In-context learning of novel functions.** Akyürek et al. 2023 and follow-ups show Transformers learn *new* algorithms in-context (linear regression on novel weights, modular arithmetic on novel moduli). The SSM line has weaker published results here; the bottleneck appears to be the same fixed-state compression that hurts recall. No strong SSM baseline exists on the canonical ICL function-class benchmarks.

4. **Compositional / algorithmic generalization.** Lake & Baroni SCAN, COGS, and dyck-grammar tasks: linear-recurrent models match Transformers in-distribution but underperform on length-extrapolation in compositional settings. Documented less rigorously than (1)–(3) but consistent.

5. **State-tracking / parity / non-regular languages.** Liu et al. (arXiv 2404.08819, "Theoretical Foundations of Deep SSMs") and Merrill, Petty, Sabharwal (arXiv 2404.08819-class results) show diagonal linear SSMs (S4D, LRU, Mamba) **cannot solve word-problems on non-solvable groups** in O(1) depth — a strict expressivity gap relative to even RNN. DeltaNet recovers some of this with non-diagonal state.

6. **Optimization fragility.** Zoology reports Mamba and Hyena have *very* narrow optimal-LR windows: performance near-zero at most LRs, suddenly near-optimal at a specific value. Transformers do not show this pathology to the same degree. Suggests the wins reported in the original papers may be partially optimization-coverage artifacts.

**Where genuine novelty is most plausible for VESTA:**

- **(2) + (5):** the *expressivity* gap is mathematically real, not an optimization artifact. A novel non-diagonal / non-LTI mechanism that beats DeltaNet on state-tracking *and* doesn't lose Mamba's throughput would be a publishable contribution.
- **R6 + R8 axes:** no strong baseline exists in the literature. A well-designed memory-write regularizer or a learned "when to decode raw tokens" gating would have no direct competitor, but the burden is on the framework to show the *task* discriminates these from R3 + standard CE.
- **R2 (latent prediction):** the literature is *empty* on a published win at iso-encoder-params for autoregressive LM. EALRMN Phase-0a already failed here once; re-attempting requires either a new task design or a substantively different latent objective.

---

## 4. C++ reference-implementation status (glades-ml)

| Method | Upstream lang | In glades-ml? | Cheapest C++ port |
|--------|---------------|---------------|-------------------|
| S4 | JAX | No | Moderate (needs HiPPO init, FFT conv) |
| S5 | JAX | No | Moderate |
| Mamba | Python+Triton | No | Hard (selective-scan kernel) |
| Mamba-2 | Python+CUDA | No | Hard |
| RWKV | Python; **rwkv.cpp exists** (saharNooby) for inference | No | Easiest — port `rwkv.cpp` inference and add training |
| LRU | JAX | No | **Easiest** — pure diagonal complex linear recurrence, ~200 LOC |
| Hyena | PyTorch | No | Moderate (FFT conv) |
| RetNet | PyTorch | No | Moderate |
| Griffin/Hawk | JAX | No | Moderate (RG-LRU is small; sliding-window attention exists in glades transformer stack already) |
| Mega | PyTorch | No | Easy (EMA + standard attention) |

The glades-ml `network.h` defines only `TYPE_RNN/GRU/LSTM` (line 2119) on the recurrent side, plus DFF/Transformer/CNN. **No SSM-class architecture exists in this codebase.** For VESTA the cheapest path to a published-quality baseline is **LRU** (200 LOC pure diagonal complex linear RNN); for a strong selective baseline it is a **Hawk-style RG-LRU** (small kernel, reuses local-attention from the existing transformer stack); a full Mamba port requires a custom selective-scan kernel and should not be undertaken without a specific reason.

---

## 5. Implications for VESTA

1. **Mandatory baseline replication (claim B0 in `newmodel.txt`)** is well-grounded: linear-recurrence + orthogonal init is a known phenomenon; reproducing it confirms the testbed. Use **LRU** as the strong-baseline reference, not a hand-rolled linear RNN, to avoid re-running the EALRMN error.

2. **R1, R5, R7 already have very strong baselines** (MambaByte, Mixtral, MoD). A novel mechanism on these axes that doesn't beat the named baselines should be discarded.

3. **R3 (recurrence) is well-saturated.** Beating Mamba-2 or Hawk by ≥0.05 nat at iso-param on long-context retrieval is unlikely without attacking documented weaknesses (recall, copying, state-tracking).

4. **R2, R6, R8 have no strong baseline.** This is where invention has the most room — *if* a task can be designed that discriminates the proposed mechanism from a trivial null. The EALRMN R2 (latent prediction) precedent (Phase-0a failure) is a warning, not a deterrent: the failure was task design, not the question.

5. **The most promising attack direction is documented weakness (2) + (5): exact copying / state-tracking expressivity.** DeltaNet partially closes this. A non-diagonal state mechanism with stronger expressivity guarantees that doesn't destroy Mamba's throughput would be a real contribution, and there is *no* well-established C++ baseline to compete against — the field is still developing.

---

## Sources

- [S4 — Efficiently Modeling Long Sequences with Structured State Spaces (Gu, Goel, Ré 2021)](https://arxiv.org/abs/2111.00396)
- [S5 — Simplified State Space Layers for Sequence Modeling (Smith, Warrington, Linderman 2022)](https://arxiv.org/abs/2208.04933)
- [Mamba — Linear-Time Sequence Modeling with Selective State Spaces (Gu, Dao 2023)](https://arxiv.org/abs/2312.00752)
- [Mamba-2 / SSD — Transformers are SSMs (Dao, Gu 2024)](https://arxiv.org/abs/2405.21060)
- [Mamba-2 algorithms blog (Tri Dao)](https://tridao.me/blog/2024/mamba2-part1-model/)
- [RWKV-4 — RWKV: Reinventing RNNs for the Transformer Era (Peng et al. 2023)](https://arxiv.org/abs/2305.13048)
- [Eagle and Finch (RWKV-5/6) (Peng et al. 2024)](https://arxiv.org/pdf/2404.05892)
- [RWKV-7 "Goose" with Expressive Dynamic State Evolution (Peng et al. 2025)](https://openreview.net/pdf?id=ayB1PACN5j)
- [LRU — Resurrecting Recurrent Neural Networks for Long Sequences (Orvieto et al. 2023)](https://arxiv.org/abs/2303.06349)
- [LRU PMLR PDF](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)
- [Hyena Hierarchy (Poli et al. 2023)](https://arxiv.org/abs/2302.10866)
- [RetNet — Retentive Network: A Successor to Transformer (Sun et al. 2023)](https://arxiv.org/abs/2307.08621)
- [Griffin / Hawk — Mixing Gated Linear Recurrences with Local Attention (De et al. 2024)](https://arxiv.org/abs/2402.19427)
- [Mega — Moving Average Equipped Gated Attention (Ma et al. 2022)](https://arxiv.org/abs/2209.10655)
- [MambaByte — Token-free Selective State Space Model (Wang et al. 2024)](https://arxiv.org/abs/2401.13660)
- [Based — Simple linear attention language models balance the recall-throughput tradeoff (Arora et al. 2024)](https://arxiv.org/abs/2402.18668)
- [DeltaNet — Parallelizing Linear Transformers with the Delta Rule (Yang, Schlag et al. 2024)](https://arxiv.org/abs/2406.06484)
- [Zoology blog post — Measuring and Improving Recall in Efficient LMs (Eyuboglu, Arora, Zhang, Ré 2023)](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)
- [Repeat After Me / Mamba Copy Limitations (Jelassi et al. 2024)](https://arxiv.org/html/2410.03810v3)
- [Mixture-of-Depths (Raposo et al. 2024)](https://arxiv.org/abs/2404.02258)
- [Mixture-of-Recursions (Bae et al. 2025)](https://arxiv.org/abs/2507.10524)
- [Based blog (Stanford Hazy Research)](https://hazyresearch.stanford.edu/blog/2024-03-03-based)
- [Mamba-2 Princeton PLI blog](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)
- [Empirical Study of Mamba-based Language Models (Waleffe et al. 2024)](https://arxiv.org/pdf/2406.07887)
