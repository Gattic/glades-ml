# SPAREC × CHIRON Incompatibility — Design Flaw Analysis

**Date:** 2026-04-23 (Ralph-loop iter 127)
**Finding:** Paradigm shift #35 SPAREC (FFN backward sparsity) does NOT apply
to CHIRON's reversible-flow architecture. Phase 1 primitives exist but there
is no trainer integration path.

---

## 1. The architectural mismatch

CHIRON's per-layer block (from `chiron_main.cpp:10-11`):
```
p += attention_shear(q ; Wq, Wk, Wv, Wo)    // symplectic shear
q  = reln(q ; gamma, beta)                  // reversible LayerNorm
```

Compare to a standard transformer block:
```
q  = q + attention(LN(q))
q  = q + FFN(LN(q))                         // ← no analogue in CHIRON
```

**CHIRON has NO FFN, no GELU, no activation sparsity to exploit.**

SPAREC's entire mechanism (threshold |σ'(x)| to identify inactive neurons
in FFN backward) is predicated on the existence of a forward
`h_in → W_up → σ(·) → W_down → h_out` path that CHIRON does not have.

## 2. Why this was missed during SPAREC design

Paradigm #35 SPAREC was designed in iteration 109-110 with full mathematical
rigor. The Phase 1 primitives (`gpu_sparec.{h,cu}`) were validated via unit
tests at bit-exact parity. But the design brief implicitly assumed a
standard transformer architecture.

The assumption was never verified against `chiron_main.cpp` — which uses only
attention + LayerNorm shear blocks.

**Lesson:** Gate-0 probes should include an ARCHITECTURE-FIT check — does
the target training path actually have the mechanism's target substructure?

## 3. Possible rescues

### Option A — Port SPAREC to standard-transformer trainer
- `glades_pile_train` may use the standard transformer with FFN
- SPAREC could wire into that trainer's FFN backward
- Verify: does glades_pile_train use FFN?

### Option B — Add an FFN to CHIRON
- Breaks CHIRON's "pure reversible shear" invariant
- FFN is NOT a symplectic shear → loses reversibility
- Defeats CHIRON's O(1)-in-depth activation memory benefit
- REJECTED on architectural grounds

### Option C — Rework SPAREC to target LN or attention output
- CHIRON has attention shear (QKV, WO) which could have row-sparse gradients
- No nonlinearity in LN (only affine), so no σ'(x) analog
- Attention's softmax IS a nonlinearity — softmax_probs near zero could be skipped in backward
- **New direction: SPAREC-ATTN**: threshold softmax probabilities to skip low-probability keys in attention backward

### Option D — Abandon SPAREC for CHIRON, keep it as a library-level contribution
- Phase 1 primitives remain valid for any future FFN-based trainer
- Document the scope limitation
- Pursue orthogonal paradigm directions

## 4. Immediate decision (iter 127)

**Chosen: Option D + Option C exploration.**

- Keep Phase 1 primitives as-is (valid for hypothetical FFN trainer)
- Document SPAREC × CHIRON incompatibility (this doc)
- Re-scope paradigm #35 next iteration to consider attention-softmax
  sparsity (SPAREC-ATTN)

## 5. Implications for the research program

The CHIRON trainer's compute decomposition:
1. Attention forward: ~40% (T² scaling dominates at T=1024)
2. Attention backward: ~35%
3. LayerNorm forward+backward: ~10%
4. Embedding gather + unembed: ~15%

**All remaining speedup opportunities are in attention or embedding.**

SPAREC's FFN-targeting mechanism is a non-starter for CHIRON. The
research program's concentration of effort should redirect to:

- Attention compute (local-window, sliding-window, top-K variants)
- Sequence-length curriculum (attacks T² scaling)
- Attention-softmax sparsity (SPAREC-ATTN, new direction)

FACE + MFIO + bf16 compound has saturated the MEMORY axis for CHIRON
at 1.84B (27× scale range). The SPEED axis remains open on the
attention front, where ~75% of per-step compute lives.

## 6. Research-methodology lesson captured

Before designing a paradigm, verify its mechanism target exists in the
concrete trainer the design is intended to accelerate. Specifically:
- Architecture fit check: do the required primitives (FFN, softmax,
  nonlinearity, etc.) exist in the trainer's actual forward pass?
- Compute proportion check: does the target axis hold ≥ 20% of step
  compute? If not, even a 10× mechanism speedup gives < 2× end-to-end.

This should be part of the Gate-0 protocol going forward.
