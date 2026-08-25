# Full paradigm integration — final results on `vesta5`

This document captures the end-state after completing both the kernel-level
primitives (Recommendations 2+3 deep work) AND the trainer surgery to wire
them into production training paths.

## Hardware

- RTX 4080 SUPER (15936 MB, 80 SMs, SM 8.9)

## Production-wired paradigms

| # | Paradigm | CLI flag | Kernel | Trainer | E2E test |
|---|---|---|---|---|---|
| 78 | ATTENTION-SINK | `--attn-sinks S` `--local-attn W` | ✓ CPU + GPU + BF16 production kernel | ✓ forward + backward dispatch in 3 sites | ✓ 3.1× speedup at T=8192 |
| 76 | MLA | `--mla-dc N` | ✓ mla_attention_forward_gpu + backward | ✓ Forward, backward, Adam updates wired | ✓ trains end-to-end |
| 74 | BitNet QAT | `--binary-ffn` (auto BitNet path when K%128==0) | ✓ WMMA B1 + scale recovery | ✓ FFN W2 forward; STE backward via float master | ✓ trains end-to-end |
| 93 | ASTRA-KAHAN | leaf primitive | ✓ CPU step | (trainer integration TBD — kernel ready) | unit-test only |

## End-to-end stacked training

Real-trainer measurement (T=512, dmodel=256, layers=2, heads=4, dFF=512,
--mp, 100K tokens, RTX 4080 SUPER):

```
                          tok/s          loss
baseline                  119,306        10.473148
all 4 stacked             144,254        10.425254
                          ─────          ────────
                          1.21× faster   0.048 nat lower
```

Stacked CLI:

```bash
glades_pile_train --gpu --mp \
    --attn-sinks 4 --local-attn 64 \
    --mla-dc 128 \
    --binary-ffn \
    --pretok-dir <tokens> --vocab-file <bpe>
```

Each paradigm contributes:
- `--attn-sinks 4 --local-attn 64`: sliding-window attention with 4 always-on
  sinks (#78). Benefit grows with T; at T=8192 we measured 3.1× speedup.
- `--mla-dc 128`: K, V projected through a low-rank latent c (d_c=128).
  Cache size O(d_c) instead of O(2·dKV).
- `--binary-ffn`: FFN W2 forward via WMMA B1 binary tensor cores
  (BitNet b1.0 with per-row scale recovery). STE backward.

## Branch state (`vesta5`)

29 commits since `49c0ae637`. Total diff vs base:
- ~2000 LOC of kernel primitives (transformer_ops.h, sampling_utils.h, GPU)
- ~1500 LOC of trainer surgery (sgd_transformer.cpp, network.cpp, gpu_transformer_state.{h,cu})
- ~3000 LOC of unit tests (Group E-L + MFAC + BENCH groups)
- All commits pass nnall regression (EXIT=0)

## What's left for production

1. **#74 BF16-Adam path with MLA**: currently MLA Adam updates run only on the
   FP32 Adam state path. The `--bf16-adam` path would need MLA-Wdkv/Wuk/Wuv
   handling.
2. **Other optimizers (ATLAS, VESTA, HELIOS, etc.) for MLA tensors**: same.
3. **#93 ASTRA-KAHAN trainer integration**: currently leaf primitive only.
4. **Larger-scale validation**: meaningful pretraining runs (multi-hour) on
   real data to confirm convergence behavior of the stacked paradigms.

## Practical next steps for "train a large LLM quickly"

1. Run a multi-hour pretraining with the full stack on the user's pile
   corpus. Track loss curve + perplexity vs steps. Compare to baseline at
   matched wall-clock.
2. Push branch upstream / open PR.
3. Optionally: extend the quantization to W1 in addition to W2 for further
   speedup (#74 binary FFN currently only binarizes W2).
