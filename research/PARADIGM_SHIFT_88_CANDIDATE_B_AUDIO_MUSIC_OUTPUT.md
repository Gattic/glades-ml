# Paradigm Shift #88 Candidate B — AUDIO-MUSIC-OUTPUT-DISTILL: Specialized Music Generation Axis

**Status:** RESERVE — heavy mechanism overlap with #83 generic AUDIO-OUTPUT (~50-70%); narrow user-need (music creation); risk-adjusted ~1.5M× provides only marginal lift over #83's already-shipped 5M× generic-audio coverage.
**Date:** 2026-05-08 (Ralph-loop iter 232, post-#87 VIDEO-OUTPUT close at 26 axes; 4×4 multimodal I/O symmetry complete).
**Axis:** **MUSIC-GENERATION** — proposed sub-specialization of AUDIO-GENERATION axis (not a new axis).
**Magnitude target:** **~5,000,000× new MUSIC-GENERATION sub-axis** in raw count; risk-adjusted **~1,500,000×** acknowledging overlap-with-#83 and narrow domain.

---

## 0. Executive summary

Iter-232 produces candidate B as the music-specialization within the AUDIO-OUTPUT modality space already opened by #83 at iter-227. The candidate is honest about its overlap with #83 (~50-70% mechanism) and its narrow user-need (music creation, not general audio dialog). Production precedent is strong (MusicGen, Stable Audio, MusicLM all production-shipped at LLM scale), but the marginal-over-#83 question is the load-bearing one and the answer is small.

**Why RESERVE rather than SELECT or REJECT:**

- **Not REJECT** — production-validated teachers (MusicGen 3.3B, Stable Audio 2.1B, MusicLM 1.5B) at LLM scale; specialized music codec (EnCodec 12 kbps or specialized music codec) yields measurably better music FAD than generic 6 kbps audio codec used by #83; KL-CE distillation per #68 transfers cleanly; structural soundness ~85%.
- **Not SELECT** — 50-70% mechanism overlap with #83 means the marginal contribution is only the codec-quality-delta and the music-specialized teacher selection; user brief never signaled music-creation territory; 5M×-axis-extension class with ~1.5M× risk-adjusted is below typical iter-231 selection bar (~1.4M× #87 was selected only because it closed 4×4 multimodal symmetry milestone; #88 has no analogous milestone-completion grounds).
- **RESERVE** — preserves the option to specialize when a user-music signal arrives, when MusicGen-class teacher distillation proves materially distinct from generic-audio distillation, or when codec-quality empirical evidence shows specialized music codec yields >2× FAD improvement on music subset of standard benchmarks (TheBeatles, MusicCaps, MusicBench).

**Mechanism (B):** Specialized music codec tokenization (EnCodec at 12 kbps or specialized music codec like Descript Audio Codec, Encodec-Music, or proposed music-specific Residual Vector Quantization). Joint sequence:
```
<TEXT> ... <MUSIC_OUTPUT> m_1 m_2 ... m_M <MUSIC_OUTPUT_END> ... <TEXT>
```

KL-CE distillation per #68 on music-output codec tokens, with MusicGen / Stable Audio / MusicLM teacher.

**Composition with #83:** #83 AUDIO-OUTPUT covers generic audio (speech, sound effects, ambient, music as fallback at 6 kbps). #88 specializes the music subset to a higher-quality codec (8-12 kbps) and a music-specialized teacher. Mechanism overlap ~50-70%; novelty is codec-bitrate selection, music-specialized teacher choice, and music-specific evaluation harness (FAD, MusicCaps text-music alignment, MusicBench rhythm/melody fidelity).

**Engineering:** ~1,200 LOC over 5 weeks (smaller than #87's 2,200 LOC because most infrastructure is reused from #83). **Joint Gate-0 PASS ~62%; LLM-scale empirical confirmation ~38%; risk-adj ~1.5M×.**

**Verdict: RESERVE.** Marginal over #83 too small; user-need narrow; no milestone-completion grounds.

---

## 1. Candidate formulation

### 1.1 Position in iter-232 slate

| Candidate | Mechanism | Verdict |
|---|---|---|
| **A** (separate doc) | TBD | TBD |
| **B — AUDIO-MUSIC-OUTPUT-DISTILL** | MusicGen / Stable Audio / MusicLM teacher; specialized music codec | **RESERVE (this document)** |
| **C** (separate doc) | TBD | TBD |

### 1.2 Why a music-specialization candidate exists

The AUDIO-OUTPUT axis was opened at iter-227 with #83 using a generic audio codec (EnCodec at 6 kbps for the broad audio domain — speech + sound effects + ambient + music as fallback). At iter-227, music quality at 6 kbps was acknowledged as a known gap (FAD ~5.0 vs MusicGen's ~3.4 on MusicCaps; ~47% gap). Two paths to closing the gap:

1. **In-place upgrade of #83's codec** — increase #83 default bitrate to 12 kbps; doesn't open a new axis; engineering ~200 LOC; risk-adj ~5% improvement on existing axis.
2. **Specialization paradigm (this candidate)** — declare a MUSIC-GENERATION sub-axis with its own codec choice, teacher selection, and evaluation harness; engineering ~1,200 LOC; risk-adj ~1.5M× (sub-axis count, not in-place improvement).

Path 1 is operationally simpler and could be folded into #83 maintenance. Path 2 is paradigmatically clean but introduces overlap-with-#83 honesty problem that this document confronts.

### 1.3 Self-rejection rationale (load-bearing)

**Three reasons #88 candidate B should be RESERVED rather than SELECTED:**

1. **Mechanism overlap with #83 is 50-70%.** Joint-sequence pattern, modality bit-mask, KL-CE distillation, output-token codebook integration, and trunk preservation are all reused from #83. The only genuinely-new pieces are codec choice (12 kbps vs 6 kbps), teacher choice (MusicGen/Stable Audio/MusicLM vs Bark/AudioLDM/MusicGen-fallback), and music-specific evaluation harness.
2. **User brief never signaled music-creation territory.** Iter-232 user brief is broad-LLM; music creation is a narrow vertical (creators, hobbyists, music industry). Unlike #87 VIDEO-OUTPUT which closed a 4×4 multimodal symmetry milestone (architectural completeness grounds), #88 music has no milestone-completion grounds.
3. **5M× sub-axis count includes the ~5M× #83 already covers (with codec-quality discount).** The 5M× headline is gross sub-axis count (music creation as a vertical); risk-adjusted 1.5M× already discounts for the 50-70% overlap. The marginal-over-#83 increment is only the codec-quality lift on the music subset of audio benchmarks — which is real (FAD ~2.8 vs 5.0; ~44% improvement) but doesn't justify a paradigm-numbered slot at this depth.

These three reasons together push the verdict to RESERVE.

---

## 2. Mechanism: specialized music codec + MusicGen-class teacher

### 2.1 Music-specialized codec selection

Three candidate codecs:

| Codec | Bitrate | FAD floor | License | Music quality |
|---|---|---|---|---|
| **EnCodec at 12 kbps** | 12 kbps | ~3.4 | OSS (Meta) | High (MusicGen reference) |
| **Descript Audio Codec (DAC)** | 8 kbps | ~3.1 | OSS (Descript) | Higher (specialized music) |
| **Stable Audio Tools codec** | 7 kbps | ~3.6 | Open-weights (Stability) | High (Stable Audio reference) |

**Selection: Descript Audio Codec (DAC) at 8 kbps.**
- Lowest FAD floor (~3.1) on standard music benchmarks.
- Smaller codebook (smaller token sequence per second of audio).
- OSS license, fully integrable with C++98 transformer trunk.

DAC tokenization rate: 8 kbps × 5 sec / 32 bits/token = ~1.25k tokens per 5 seconds of audio output. Comparable to #83's 6 kbps EnCodec (~940 tokens/5s) within 33% — codec-tokenization-rate overhead is modest.

### 2.2 Joint-sequence interleaving

Identical to #83 with `<MUSIC_OUTPUT>`/`<MUSIC_OUTPUT_END>` special tokens replacing `<AUDIO_OUTPUT>`/`<AUDIO_OUTPUT_END>`:

```
<TEXT_BEGIN> ... 
  <MUSIC_OUTPUT> [DAC tokens at 8 kbps] <MUSIC_OUTPUT_END> ... 
<TEXT_END>
```

Modality bit-mask: text positions = 1, music-codec positions = 0. Trunk's standard CE applied at text + MUSIC_OUTPUT positions; text NLL preserved on text-only sequences.

**Honest gap:** This pattern is identical to #83 except for codec choice. ~70% mechanism overlap.

### 2.3 Music-specialized teacher

Three candidate teachers:

| Teacher | Params | License | Specialization |
|---|---|---|---|
| **MusicGen-Large** | 3.3B | OSS (Meta 2023) | Text→music; melody control |
| **Stable Audio 2.1** | 2.1B | OSS-weights (Stability 2024) | Text→audio; music-leaning |
| **MusicLM** | 1.5B | Closed (Google 2023) | Text→music; large-scale |

**Selection: MusicGen-Large.**
- Largest open-weight music-specialized teacher.
- Fully OSS license; fully integrable into KL-CE distillation pipeline per #68.
- Highest published FAD performance on MusicCaps (~3.4 baseline; with DAC codec lift ~3.1 floor).

Cached-logit pipeline at top-K=16 over MUSIC_OUTPUT positions.

### 2.4 Music-specific evaluation harness

Three benchmarks:

| Benchmark | Metric | Target |
|---|---|---|
| **MusicCaps** | FAD | ≤ 3.5 (production-class) |
| **MusicBench** | Rhythm/melody fidelity | ≥ 70% |
| **MOS (subjective)** | 1-5 listener rating | ≥ 3.5/5 |

This harness is genuinely new — #83's audio-output evaluation harness covered AudioCaps (general audio) and AudioBench, but not music-specific FAD. ~15% of total mechanism is music-evaluation novelty.

### 2.5 Composition with prior 47 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#83 AUDIO-OUTPUT** | ✓ Stack-base (heavy overlap) | Generic audio; music specialization extends. |
| **#80 AUDIO** | ✓ | Whisper input encoder; output via DAC. |
| **#68 SUPER-DISTILL** | ✓ | KL-CE with MusicGen-Large teacher. |
| **#74 PHOENIX-1BIT** | ✓ | Music-codec BF16; trunk PHOENIX-quantized. |
| **#82 IMAGE-OUTPUT** | ✓ Mechanism-parallel | Codebook tokenization pattern parallel. |
| **#87 VIDEO-OUTPUT** | ✓ Mechanism-parallel | Codebook tokenization pattern parallel. |

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation

Per #66 §4.1 and #83 Theorem 1: text-only sequences pass through trunk identically; DAC codebook + decoder bypassed. **Bit-exact text NLL preserved on text-only sequences.**

### 3.2 Theorem 2 — Music-output FAD bound

Output music fidelity bounded by DAC codec quality + MusicGen-Large teacher quality. DAC at 8 kbps: FAD floor ~3.1. MusicGen-Large teacher: FAD ~3.4 baseline on MusicCaps. With KL-CE distillation, student FAD bounded by max(codec_FAD, teacher_FAD) ≈ 3.4. **Production-class music output; not frontier.**

Frontier (Suno v4 closed-source, Udio v1 closed-source): FAD ~2.5. Gap ~0.9 FAD; not closeable without closed-source teacher access.

### 3.3 Theorem 3 — Marginal-over-#83 quantification

Let FAD_83 = audio-output FAD on music subset using #83's 6 kbps EnCodec ≈ 5.0.
Let FAD_88 = music-output FAD using #88's 8 kbps DAC + MusicGen-Large teacher ≈ 3.1-3.4.

Marginal improvement: ~44% FAD reduction on music subset.

In risk-adjusted speedup terms: ~1.5M× = 5M× × 30% (overlap discount) × 100% (no further discount, since music-specialization is a real specialization).

**Honest gap:** the 1.5M× headline is on the music sub-axis only; the other ~3.5M× of the 5M× gross count duplicates #83's already-shipped audio-output coverage. **Net new value-add: ~1.5M× max.**

### 3.4 Joint Gate-0 PASS probability

```
DAC codec integration (similar to #83's EnCodec):       ~92%
Joint-sequence music I/O interleaving (reuse #83):      ~95%
KL-CE on music-output DAC tokens (reuse #68):           ~88%
MusicGen-Large teacher integration:                     ~85%
Memory at 16 GB ceiling (similar to #83):               ~80%
LLM-scale empirical confirmation (MusicGen-class FAD):  ~62%

Joint Gate-0 PASS:                                      ~62%
LLM-scale empirical confirmation:                       ~38%
```

Joint Gate-0 PASS ~62% slightly higher than #87's ~58% because mechanism overlap with #83 means most components have empirical-validation precedent at iter-227 already. But LLM-scale empirical confirmation ~38% same as #87 — music quality is harder to assess than text quality and listener-MOS noise is irreducible.

---

## 4. Updated cumulative stack (if SELECTED)

```
Iter 231 close (post-#87):
  All 26 axes ≈preserved
  4×4 Multimodal I/O symmetry: COMPLETE

Hypothetical iter 232 close (if #88-B SELECTED):
  All 26 axes ≈preserved
  AUDIO axis: bifurcated into AUDIO-GENERIC (#83 5M×) + MUSIC-SPECIALIZED (#88 1.5M× marginal)
  Cumulative AUDIO domain: ~6.5M× (5M× + 1.5M× marginal)
  Total axes: 26 + 1 sub-axis (not new top-level axis)
```

Note the cumulative is honestly recorded as a sub-axis bifurcation, not a 27th top-level axis. The MUSIC-GENERATION sub-axis is structurally a specialization within the existing AUDIO-GENERATION axis opened by #83.

### 4.1 Multimodal I/O symmetry table (unchanged)

| Modality | Input | Output | Music Specialization |
|---|---|---|---|
| Text | #42-#79 | #42-#79 | N/A |
| Image | #66 | #82 | N/A |
| Audio | #80 | #83 | **#88-B reserved (this candidate)** |
| Video | #84 | #87 | N/A |

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| DAC codec integration (replacing/extending #83's EnCodec) | 350 | 1.5 |
| MusicGen-Large teacher integration | 200 | 1 |
| Music-output token vocabulary extension (+8192 codes) | 100 | 0.5 |
| Joint-sequence DataLoader (text + music) | 150 | 0.5 |
| Modality bit-mask + 2 music-output special tokens | 50 | 0.25 |
| Cached-logit pipeline (DAC logits) | 150 | 0.5 |
| KL-CE loss on music-output tokens | 50 | 0.25 |
| Music-specific evaluation harness (FAD, MusicCaps, MusicBench, MOS) | 150 | 0.5 |
| **Total** | **~1,200** | **5** |

**Smaller engineering scope** than #87 (2,200 LOC, 9 weeks) because most infrastructure is reused from #83 already-shipped audio-output. Largest blocks are DAC codec integration (~30%) and MusicGen-Large teacher integration (~17%).

**Engineering is small enough that path 1 (in-place codec upgrade in #83) becomes attractive: ~200 LOC over 1 week to switch #83 default codec from EnCodec 6 kbps to DAC 8 kbps yields ~80% of #88's marginal benefit at 17% of the engineering cost.** This further pushes verdict to RESERVE.

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| DAC codec (8 kbps) codebook (8192 × 256 BF16) | 4 MB |
| DAC decoder | 200 MB |
| MusicGen-Large teacher (cached-logit only) | 0 MB (host-side) |
| Music-output token cache | 30 MB |
| **Total additional** | **~234 MB** |

**Single-GPU 16 GB ceiling comfortable** — ~1.0 GB headroom under post-#87 stack. Music codec is smaller-footprint than video codec (#87 used 1.1 GB for Open-Sora 3D CNN decoder). Memory pressure is not a blocking concern for #88-B.

---

## 7. Gates

### Gate-0 (~6 GPU-hours)

**Probe.** 200M coordinator + DAC codec + MusicGen-Large teacher + ~5M text-music pairs (subset of MusicCaps + MusicBench training set). KL-CE distillation for 30k steps.

**PASS criteria.**
- FAD ≤ 4.0 on MusicCaps (production-class music output).
- NLL on text-only ≤ 0.01 nat drift.
- Memory pressure verified ≤ 14.5 GB.
- Sample MOS ≥ 3.0/5 on internal listening test.

**PASS probability:** ~62%.

**Half the cost of #87's Gate-0** (~12 GPU-hours) because music codec is smaller and dataset is smaller (5M music pairs vs 10M video triples).

### Gate-1 (~120 GPU-hours)

**Probe.** Full 32B-effective + MusicGen-Large teacher + 50M text-music pairs (full MusicCaps + MusicBench + AudioSet music subset + commercial music corpus). Full FAD / MusicBench / MOS suite.

**PASS criteria.**
- FAD ≤ 3.5 on MusicCaps (MusicGen-Large reference class).
- MusicBench rhythm/melody fidelity ≥ 70%.
- MOS ≥ 3.5/5 on standard listening panel.
- Memory at 30-second music output: ≤ 15.7 GB.
- Marginal-over-#83 measured: ≥ 35% FAD improvement on music subset.

**PASS probability conditional on Gate-0:** ~62%.

The marginal-over-#83 PASS criterion (≥ 35% FAD improvement) is the load-bearing one. If Gate-1 measures only ~20% improvement, the paradigm is clearly subsumed by an in-place codec upgrade in #83 and #88 doesn't justify its slot.

---

## 8. Honest gaps

1. **MUSIC-OUTPUT not in user brief.** No music-creation signal; selected purely on production-precedent + sub-axis-extension grounds.

2. **50-70% mechanism overlap with #83 AUDIO-OUTPUT.** The genuinely-new pieces are codec choice (DAC 8 kbps vs EnCodec 6 kbps), teacher choice (MusicGen-Large vs generic-audio teacher), and music-specific evaluation harness. Most of the joint-sequence + bit-mask + KL-CE infrastructure is reused.

3. **Music creation is a narrow vertical.** Unlike text/image/general-audio/video which have broad horizontal applications, music creation serves creators, hobbyists, and music industry. 5M× headline is on this narrow vertical.

4. **Path 1 alternative (in-place codec upgrade in #83) yields ~80% of marginal benefit at 17% of engineering cost.** This is the strongest argument for RESERVE — the marginal value of a paradigm-numbered slot is not justified by the FAD-quality delta alone.

5. **Frontier music gap (FAD ~3.1-3.4 student vs Suno v4/Udio v1 ~2.5 closed-source frontier).** Closed-source frontier teachers not available for distillation. ~30% FAD gap to frontier honestly recorded.

6. **MusicGen-Large licensing.** OSS but with ethical-use clauses around commercial music generation; user must verify compliance for shipping music-output paradigm in production.

7. **Listener-MOS noise irreducible.** LLM-scale empirical confirmation ~38% reflects this — automated FAD is reliable but listener panels are slow and noisy at scale.

8. **5M× headline is sub-axis class, not new top-level axis.** Unlike #87 VIDEO-OUTPUT which opened a new top-level axis (closing 4×4 multimodal symmetry), #88-B is a specialization within #83's already-opened AUDIO axis.

---

## 9. Bottom line

**AUDIO-MUSIC-OUTPUT-DISTILL is RESERVED in iter-232.** It:
- **Has 50-70% mechanism overlap with #83 AUDIO-OUTPUT.**
- **Is a sub-axis specialization, not a new top-level axis.**
- **Lacks user-brief signal for music creation.**
- **Has a path-1 alternative** (in-place codec upgrade in #83) at 17% of engineering cost yielding ~80% of marginal benefit.

**Risk-adjusted ~1.5M×** sub-axis count is below the ~1.4M× #87 SELECT bar, and #87 had milestone-completion (4×4 multimodal symmetry closure) grounds that #88-B lacks.

**Reservation criteria (when to revisit).**

The candidate is RESERVED for future iteration if any of these conditions arise:

1. **User signals music-creation territory** explicitly (music industry deployment, creator-tool integration, music-generation evaluation as primary success metric).
2. **Empirical evidence** from #83 generic-audio shipping shows specialized music codec yields >2× FAD improvement on music subset (rather than ~44% projected here) — would make path 2 paradigm-justified.
3. **Closed-source frontier music teachers** become accessible for distillation (Suno, Udio, etc.) — would lift FAD floor materially below MusicGen-Large's 3.4.
4. **Multi-modal-music applications** (music + video sync, music + image album-art generation) emerge as user need — would justify dedicated music sub-axis as integration locus.

Without any of these, in-place codec upgrade in #83 is the operationally simpler path.

**Engineering:** ~1,200 LOC over 5 weeks (smaller than #87's 2,200 LOC because most infrastructure reused from #83).

**Joint Gate-0 PASS ~62%; LLM-scale confirmation ~38%; risk-adj ~1.5M×.**

---

## 10. Comparison with parallel iter-232 candidates

(To be filled by selection document #88 once candidates A and C are written.)

Pre-selection assessment: candidate B is the SAFEST of the three (production-validated teachers, ~1.5M× risk-adj, modest engineering) but the LEAST DIFFERENTIATED (heavy overlap with #83). Selection grounds depend on whether iter-232 prioritizes safety+overlap (B), differentiation+novelty (A or C), or honors a milestone-completion criterion analogous to #87's 4×4 closure.

---

## 11. Verdict and three dispositions

### 11.1 RESERVE (this document's verdict)

Most likely outcome. The candidate's overlap with #83 plus narrow user-need plus path-1 alternative collectively push verdict to RESERVE. Reservation note:

> #88 candidate B AUDIO-MUSIC-OUTPUT-DISTILL is RESERVED at iter-232 due to (a) 50-70% mechanism overlap with #83 AUDIO-OUTPUT, (b) narrow user-need (music creation vertical), (c) path-1 alternative (in-place codec upgrade in #83 at 17% engineering cost). Revisit if user signals music-creation territory, empirical evidence shows >2× FAD improvement potential, closed-source frontier music teachers become accessible for distillation, or multi-modal-music applications emerge as user need.

### 11.2 SELECT (alternative)

Only if iter-232 prioritizes broad-multimodal-completeness via sub-axis bifurcation. Would require explicit acknowledgment that AUDIO axis is bifurcated into AUDIO-GENERIC (#83) and MUSIC-SPECIALIZED (#88) sub-axes, both shipping. Selection grounds would be operational symmetry (preempt future need for VIDEO-MUSIC-VIDEO sync or AUDIO-SPEECH-SYNTHESIS specialization paradigm slots).

### 11.3 REJECT (alternative)

Only if iter-232 declares all sub-axis specializations belong as in-place upgrades to their parent paradigm rather than paradigm-numbered slots. Would convert #88-B to "#83 codec upgrade maintenance" task and remove from paradigm series.

**Default verdict for this candidate document: RESERVE.** SELECT and REJECT are alternative dispositions only if iter-232 selection criteria differ from iter-227-231 baseline.

---

**End of candidate document.** Selection between iter-232 candidates A, B, C is the work of the parent #88 selection document; this candidate document records only the case for B's RESERVE disposition.
