# CHIRON PACT — M0 Measurement Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the gated cancellation-energy share of the PIED flagship checkpoint
(`chiron_1B_pied_e4.final`) via an env-gated, eval-only probe in the trainer forward, and issue
the pre-registered M0 GO/KILL decision (kill: gated share < 3%).

**Architecture:** Host-side probe inside `glades-trainer/trainer/chiron_main.cpp` (the single-file
trainer). During eval forwards (`isTraining=false`, PIED inactive ⇒ clean increments), download
each layer's SCFA increment streams (`W.scfa_ypar` + `W.scfa_yperp`, FP32 [T×m]) right after the
layer's attention forward, accumulate the damped-sum/mass/energy fields A, M, SS on host, then
compute the PACT gate χ, the gated excess share, per-layer sign-opposition occupancy, and an
A-vs-p F2 sanity check. No glades-ml (lib) changes; no new kernels; no training-path change.

**Tech Stack:** C++98, existing `GpuBuffer<float>::download`, `getenv` gating
(`CHIRON_DEAD_ATTN_PROBE` precedent at `chiron_main.cpp:3990`), `log_info("chiron", ...)` logging
(`[sorc]`/`[sira]` precedent). Repo: `~/dev/glades-trainer`, branch `chiron4`.

**Scope:** This plan covers **M0 only** (spec §13, `docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md`).
E0–E4 (kernels, flags, gates) are contingent on M0 GO and get their own plan. This plan produces
a self-contained deliverable: the probe, the measurement, the decision record.

## Global Constraints

- Design doc (source of truth for formulas): `~/dev/glades-ml/docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md` §5.2, §13 (M0).
- Probe formulas (per token t, channel i; L=24, T=16384, m=2048):
  `A = Σ_l D_l·u_l`, `M = Σ_l D_l·|u_l|`, `SS = Σ_l u_l²`, with `u_l = ypar_l + yperp_l` (clean, eval),
  `D_{l,i} = Π_{l'≥l} cos(θ_{l',i})`, `θ = rotThetaMax·tanh(φ_l[i])` (include own layer; the A-vs-p
  check validates the convention), `‖D‖²_i = Σ_l D²_{l,i}`,
  `σ̂²_i = (1/(L·T))·Σ_t SS_{t,i}`, `χ = (M²−A²)/(M²+ε_M·‖D‖²_i·(σ̂²_i+ε₀))` with ε_M=1, ε₀=1e-12,
  `excess = max(0, SS − A²/‖D‖²_i)`, **gated share = Σ_{t,i} χ·excess / Σ_{t,i} SS**.
- Kill bar (pre-registered): gated share < 3% ⇒ KILL the PACT arc.
- The probe MUST be env-gated (`CHIRON_PACT_PROBE`), eval-only (`!isTraining`), and dead code when
  the env var is unset — zero effect on any training or val math (it only reads buffers).
- MUST NOT overwrite `chiron_1B_pied_e4.final` (the run may write `wideval_scratch.final` — that is
  a throwaway). Verify the flagship checkpoint mtime is unchanged after the run.
- C++98 only (no auto, no range-for, no lambdas, no nullptr — use 0/NULL; `std::vector`, `double`
  accumulators are fine).
- Trainer builds with `bash build.sh` from `~/dev/glades-trainer`. No glades-ml changes ⇒ no
  `make install` needed.
- All commits on `chiron4` in `~/dev/glades-trainer`; commit messages end with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Host RAM budget: probe uses ~3.6 GB (24×[T×m] FP32 layer cache + 3×[T×m] fields). Check
  `free -g` ≥ 6 GB available before the run; if short, set `CHIRON_PACT_PROBE_MAX` lower (the
  cache is per-sequence, reused).

---

### Task 1: Probe state, env gating, damping table, and startup self-test

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` (three insertion points: file-static state
  near the top-level static helpers ~line 11900 region, next to `log_sira_terminal_diagnostics`;
  env parse next to the `CHIRON_DEAD_ATTN_PROBE` getenv at ~line 3990; no hooks yet)

**Interfaces:**
- Produces (used by Task 2): `struct PactProbeState` + global `g_pactProbe`;
  `static bool pact_probe_build_damp(const Config& cfg, ChironParams& W);`
  `static void pact_probe_selftest();`
  Fields: `bool enabled; int maxSeq; int seqDone; bool dReady; bool seqActive;`
  `std::vector<float> D, Dsq, A, Mm, SS; std::vector< std::vector<float> > uCache;`
  aggregates `double aggSS, aggExcess, aggGatedExcess; long chiHist[10];`
  `std::vector<double> occNum, occDen;` and scratch `std::vector<float> hostYpar, hostYperp, hostP;`
- Consumes: `Config` fields `cfg.L, cfg.m, cfg.T, cfg.rotThetaMax, cfg.rotCoupling`;
  `ChironParams::rot_phi` (`std::vector<glades::gpu::GpuBuffer<float>*>`, per-layer length m);
  `GpuBuffer<float>::download(float*, size_t)`.

- [ ] **Step 1: Add the probe state and helpers** (place near the other static diagnostics helpers,
  e.g. just above `log_sira_terminal_diagnostics` at ~line 11910):

```cpp
// ---------------------------------------------------------------------------
// PACT M0 probe (env-gated, eval-only). Measures the gated cancellation-energy
// share of the increment assembly on the p-bus. Design + pre-registered kill
// bar: glades-ml docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md
// (S5.2 formulas, S13 M0 gate: KILL if gated share < 3%).
// Enable: CHIRON_PACT_PROBE=1 [CHIRON_PACT_PROBE_MAX=N] on an eval run.
// ---------------------------------------------------------------------------
struct PactProbeState {
    bool enabled;
    int maxSeq;      // sequences to probe (env CHIRON_PACT_PROBE_MAX, default 8)
    int seqDone;
    bool dReady;     // damping table built
    bool seqActive;  // capture in progress for the current forward()
    std::vector<float> D;    // [L*m] D[l*m+i] = prod_{l'>=l} cos(theta_{l',i})
    std::vector<float> Dsq;  // [m]   sum_l D^2
    std::vector<float> A;    // [T*m] sum_l D_l * u_l
    std::vector<float> Mm;   // [T*m] sum_l D_l * |u_l|
    std::vector<float> SS;   // [T*m] sum_l u_l^2
    std::vector< std::vector<float> > uCache; // L x [T*m] (per-layer u, for occupancy)
    std::vector<float> hostYpar, hostYperp, hostP; // [T*m] download scratch
    double aggSS, aggExcess, aggGatedExcess;
    long chiHist[10];
    std::vector<double> occNum, occDen; // [L] opposing-mass occupancy
    PactProbeState() : enabled(false), maxSeq(8), seqDone(0), dReady(false),
                       seqActive(false), aggSS(0.0), aggExcess(0.0), aggGatedExcess(0.0) {
        for (int b = 0; b < 10; ++b) chiHist[b] = 0;
    }
};
static PactProbeState g_pactProbe;

// Damping table from the loaded rot_phi (device) — one download per layer.
// Convention: the layer-l increment passes through Phi_l..Phi_{L-1} (shear
// precedes WhiSC within a layer), so D includes the OWN layer's cos(theta).
// The A-vs-p sanity check in pact_probe_finalize validates this in vivo.
static bool pact_probe_build_damp(const Config& cfg, ChironParams& W) {
    const int L = cfg.L, m = cfg.m;
    g_pactProbe.D.assign((size_t)L * (size_t)m, 1.0f);
    g_pactProbe.Dsq.assign((size_t)m, 0.0f);
    if (cfg.rotCoupling && !W.rot_phi.empty()) {
        std::vector<float> phi((size_t)m);
        std::vector<double> suffix((size_t)m, 1.0); // prod_{l'>l} cos(theta_{l'})
        for (int l = L - 1; l >= 0; --l) {
            if (!W.rot_phi[(size_t)l]->download(&phi[0], (size_t)m)) return false;
            for (int i = 0; i < m; ++i) {
                const double th = (double)cfg.rotThetaMax * tanh((double)phi[i]);
                const double c = cos(th);
                g_pactProbe.D[(size_t)l * m + i] = (float)(suffix[(size_t)i] * c);
                suffix[(size_t)i] *= c;
            }
        }
    }
    for (int l = 0; l < L; ++l)
        for (int i = 0; i < m; ++i) {
            const float d = g_pactProbe.D[(size_t)l * m + i];
            g_pactProbe.Dsq[(size_t)i] += d * d;
        }
    g_pactProbe.dReady = true;
    return true;
}

// Hand-checked toy self-test (runs once at startup when the probe is enabled).
// Toy 1: L=2,T=1,m=1, u={+1,-1}, D=1: A=0, M=2, SS=2, |D|^2=2, sigma2=1,
//        chi=(4-0)/(4+1*2*1)=2/3, excess=2, gated share=(2/3*2)/2=2/3.
// Toy 2: u={+1,+1}: A=2, M=2 -> chi=0, gated share=0.
static void pact_probe_selftest() {
    const double eps0 = 1e-12;
    double u1[2] = { 1.0, -1.0 };
    double u2[2] = { 1.0,  1.0 };
    int fails = 0;
    for (int c = 0; c < 2; ++c) {
        const double* u = (c == 0) ? u1 : u2;
        double A = u[0] + u[1];
        double Mm = fabs(u[0]) + fabs(u[1]);
        double SS = u[0] * u[0] + u[1] * u[1];
        double Dsq = 2.0, sigma2 = SS / 2.0;
        double chi = (Mm * Mm - A * A) / (Mm * Mm + 1.0 * Dsq * (sigma2 + eps0));
        double excess = SS - A * A / Dsq; if (excess < 0.0) excess = 0.0;
        double share = (chi * excess) / SS;
        double expectShare = (c == 0) ? (2.0 / 3.0) : 0.0;
        if (fabs(share - expectShare) > 1e-9) ++fails;
    }
    if (fails) log_info("chiron", "[pact-m0] SELFTEST FAIL (%d) — do not trust this probe\n", fails);
    else       log_info("chiron", "[pact-m0] SELFTEST PASS\n");
}
```

- [ ] **Step 2: Add env parsing at startup.** Next to the `CHIRON_DEAD_ATTN_PROBE` getenv
  (~line 3990) — same function, immediately after it:

```cpp
    g_pactProbe.enabled = (getenv("CHIRON_PACT_PROBE") != 0);
    if (g_pactProbe.enabled) {
        const char* pmax = getenv("CHIRON_PACT_PROBE_MAX");
        g_pactProbe.maxSeq = pmax ? atoi(pmax) : 8;
        if (g_pactProbe.maxSeq < 1) g_pactProbe.maxSeq = 1;
        log_info("chiron", "[pact-m0] probe ENABLED maxSeq=%d (eval forwards only)\n",
                 g_pactProbe.maxSeq);
        pact_probe_selftest();
    }
```

  Note: if ~line 3990 sits above the helper definitions, either move the env parse to just after
  the helpers are visible, or forward-declare `static void pact_probe_selftest();` — match the
  file's existing static-declaration style.

- [ ] **Step 3: Build**

Run: `cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -5`
Expected: successful link of `glades_pile_train`, no warnings about the new code.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-trainer && git add trainer/chiron_main.cpp && \
git commit -m "pact-m0: probe state, env gating, damping table, selftest (no hooks yet)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Capture + finalize hooks in the eval forward

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` (two hooks in `forward()`: per-layer
  capture right after the SCFA dispatch at ~line 13084; finalize right after the layer loop closes,
  before the `--fuse-attn` fold at ~line 13513)

**Interfaces:**
- Consumes (from Task 1): `g_pactProbe`, `pact_probe_build_damp`.
- Consumes (existing): `W.scfa_ypar`, `W.scfa_yperp` (FP32 [T×m], valid right after
  `scfa_attention_forward(cfg, W, s, l, false, stepForDrift)` returns), `s.p` (FP32 [T×m]),
  `Tm = (size_t)T*(size_t)m`, layer loop `for (int l = 0; l < L; ++l)` at ~line 13005,
  `forward(..., bool isTraining, ...)` at line 12857.
- Produces: `[pact-m0] seq=...` per-sequence log lines and a final `[pact-m0] AGGREGATE ...` line
  (parsed in Task 3/4).

- [ ] **Step 1: Add the two probe functions** (below `pact_probe_selftest` from Task 1):

```cpp
// Per-layer capture: download the clean increment streams and accumulate.
static bool pact_probe_layer_capture(const Config& cfg, ChironParams& W, Scratch& s, int l) {
    const int T = cfg.T, m = cfg.m, L = cfg.L;
    const size_t Tm = (size_t)T * (size_t)m;
    PactProbeState& P = g_pactProbe;
    if (l == 0) {
        if (!P.dReady && !pact_probe_build_damp(cfg, W)) return false;
        if (P.A.size() != Tm) {
            P.A.assign(Tm, 0.0f); P.Mm.assign(Tm, 0.0f); P.SS.assign(Tm, 0.0f);
            P.hostYpar.resize(Tm); P.hostYperp.resize(Tm); P.hostP.resize(Tm);
            P.uCache.resize((size_t)L);
            for (int ll = 0; ll < L; ++ll) P.uCache[(size_t)ll].resize(Tm);
            P.occNum.assign((size_t)L, 0.0); P.occDen.assign((size_t)L, 0.0);
        } else {
            for (size_t j = 0; j < Tm; ++j) { P.A[j] = 0.0f; P.Mm[j] = 0.0f; P.SS[j] = 0.0f; }
        }
        P.seqActive = true;
    }
    if (!P.seqActive) return true;
    if (!W.scfa_ypar.download(&P.hostYpar[0], Tm)) return false;
    if (!W.scfa_yperp.download(&P.hostYperp[0], Tm)) return false;
    const float* Dl = &P.D[(size_t)l * (size_t)m];
    float* uc = &P.uCache[(size_t)l][0];
    for (size_t j = 0; j < Tm; ++j) {
        const float u = P.hostYpar[j] + P.hostYperp[j];
        const float d = Dl[j % (size_t)m];
        uc[j] = u;
        P.A[j]  += d * u;
        P.Mm[j] += d * (u < 0.0f ? -u : u);
        P.SS[j] += u * u;
    }
    return true;
}

// Finalize one probed sequence: gate, shares, occupancy, F2 sanity check, logs.
static void pact_probe_finalize(const Config& cfg, Scratch& s) {
    const int T = cfg.T, m = cfg.m, L = cfg.L;
    const size_t Tm = (size_t)T * (size_t)m;
    PactProbeState& P = g_pactProbe;
    if (!P.seqActive) return;
    P.seqActive = false;
    const double eps0 = 1e-12, epsM = 1.0;
    // per-channel sigma^2 = mean over (l,t) of u^2 = (1/(L*T)) * sum_t SS[t,i]
    std::vector<double> sigma2((size_t)m, 0.0);
    for (size_t j = 0; j < Tm; ++j) sigma2[j % (size_t)m] += (double)P.SS[j];
    for (int i = 0; i < m; ++i) sigma2[(size_t)i] /= ((double)L * (double)T);
    double sSS = 0.0, sEx = 0.0, sGEx = 0.0;
    long hist[10]; for (int b = 0; b < 10; ++b) hist[b] = 0;
    for (size_t j = 0; j < Tm; ++j) {
        const size_t i = j % (size_t)m;
        const double a = (double)P.A[j], mm = (double)P.Mm[j], ss = (double)P.SS[j];
        const double dsq = (double)P.Dsq[i];
        double chi = (mm * mm - a * a) / (mm * mm + epsM * dsq * (sigma2[i] + eps0));
        if (chi < 0.0) chi = 0.0; if (chi > 1.0) chi = 1.0;
        double excess = ss - a * a / dsq; if (excess < 0.0) excess = 0.0;
        sSS += ss; sEx += excess; sGEx += chi * excess;
        int b = (int)(chi * 10.0); if (b > 9) b = 9;
        ++hist[b];
    }
    // per-layer opposing-mass occupancy vs final A
    for (int l = 0; l < L; ++l) {
        const float* uc = &P.uCache[(size_t)l][0];
        const float* Dl = &P.D[(size_t)l * (size_t)m];
        double on = 0.0, od = 0.0;
        for (size_t j = 0; j < Tm; ++j) {
            const double w = (double)Dl[j % (size_t)m] * fabs((double)uc[j]);
            od += w;
            if ((double)uc[j] * (double)P.A[j] < 0.0) on += w;
        }
        P.occNum[(size_t)l] += on; P.occDen[(size_t)l] += od;
    }
    // F2 sanity: A should match p (post-stack) up to the O(sin theta) q-leak
    double num = 0.0, den = 0.0, relErr = -1.0;
    if (s.p.download(&P.hostP[0], Tm)) {
        for (size_t j = 0; j < Tm; ++j) {
            const double d = (double)P.A[j] - (double)P.hostP[j];
            num += d * d; den += (double)P.hostP[j] * (double)P.hostP[j];
        }
        relErr = (den > 0.0) ? sqrt(num / den) : -1.0;
    }
    P.aggSS += sSS; P.aggExcess += sEx; P.aggGatedExcess += sGEx;
    for (int b = 0; b < 10; ++b) P.chiHist[b] += hist[b];
    ++P.seqDone;
    log_info("chiron",
        "[pact-m0] seq=%d/%d gated_share=%.4f%% ungated_share=%.4f%% A_vs_p_relerr=%.3g\n",
        P.seqDone, P.maxSeq, 100.0 * sGEx / (sSS > 0.0 ? sSS : 1.0),
        100.0 * sEx / (sSS > 0.0 ? sSS : 1.0), relErr);
    if (relErr > 0.1)
        log_info("chiron", "[pact-m0] WARNING A-vs-p relerr %.3g > 0.1 — D convention or "
                           "probe placement suspect; measurement untrusted\n", relErr);
    if (P.seqDone >= P.maxSeq) {
        log_info("chiron", "[pact-m0] AGGREGATE gated_share=%.4f%% ungated_share=%.4f%% "
                           "(kill bar: gated < 3%%)\n",
                 100.0 * P.aggGatedExcess / (P.aggSS > 0.0 ? P.aggSS : 1.0),
                 100.0 * P.aggExcess / (P.aggSS > 0.0 ? P.aggSS : 1.0));
        { char hb[256]; int off = 0;
          for (int b = 0; b < 10; ++b)
              off += snprintf(hb + off, sizeof(hb) - (size_t)off, " %ld", P.chiHist[b]);
          log_info("chiron", "[pact-m0] chi_hist(0.0..1.0):%s\n", hb); }
        for (int l = 0; l < L; ++l)
            log_info("chiron", "[pact-m0] occ layer=%2d opposing_mass=%.4f%%\n", l,
                     100.0 * P.occNum[(size_t)l] / (P.occDen[(size_t)l] > 0.0 ? P.occDen[(size_t)l] : 1.0));
        std::fflush(stdout);
        P.enabled = false; // done; stop capturing
    }
}
```

  C++98 note: if `snprintf` is unavailable under the file's flags, use `sprintf` with the same
  bounded loop (10 entries × ≤ 21 chars fits 256). Match whichever the file already uses.

- [ ] **Step 2: Splice the per-layer capture hook.** At the SCFA dispatch in `forward()`
  (~line 13084), extend the else-if chain so capture runs only when SCFA succeeded, eval-only:

```cpp
        else if (!scfa_attention_forward(cfg, W, s, l, /*invert=*/false, stepForDrift)) return false;
        else if (g_pactProbe.enabled && !isTraining && g_pactProbe.seqDone < g_pactProbe.maxSeq
                 && !pact_probe_layer_capture(cfg, W, s, l)) {
            log_info("chiron", "[pact-m0] capture failed at layer %d; probe disabled\n", l);
            g_pactProbe.enabled = false;
        }
```

  (Adapt to the exact shape of the existing chain — the added branch must execute after a
  *successful* SCFA forward for layer `l` and must not disturb the failure `return false`.)

- [ ] **Step 3: Splice the finalize hook.** Immediately after the layer loop's closing brace
  (before the `--fuse-attn` fold block at ~line 13513):

```cpp
    if (g_pactProbe.enabled && !isTraining && g_pactProbe.seqActive)
        pact_probe_finalize(cfg, s);
```

- [ ] **Step 4: Build**

Run: `cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -5`
Expected: clean build.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-trainer && git add trainer/chiron_main.cpp && \
git commit -m "pact-m0: eval-forward capture + finalize hooks (env-gated)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Smoke run (1 sequence) against the flagship checkpoint

**Files:**
- No code changes. Produces: `~/dev/glades-trainer/logs/pact_m0_smoke_<ts>.log`

**Interfaces:**
- Consumes: the wideval eval-only protocol (resume `chiron_1B_pied_e4.final`, `--lr 0
  --whisc-ema 1.0 --no-resume-warmup --val-every 1`, per
  `logs/pied_e4_wideval_20260703_0440.log`), run.sh flagship flag routing.

- [ ] **Step 1: Pre-flight.** Verify GPU free, RAM headroom, and identify the save-name mechanism:

```bash
nvidia-smi --query-gpu=memory.used --format=csv && free -g | head -2
grep -n "wideval_scratch\|save" ~/dev/glades-trainer/run.sh | head -20
stat -c '%y %n' ~/dev/glades-trainer/database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final
```

Expected: GPU < 2 GiB used; ≥ 6 GB free RAM; find the run.sh flag that made the prior wideval run
save to `wideval_scratch` (use it verbatim below as `<SAVE_SCRATCH_FLAGS>`); record the flagship
checkpoint mtime.

- [ ] **Step 2: Smoke run — 1 probed sequence, 1 val batch.** Replicate the wideval invocation
  (compare the emitted header against `logs/pied_e4_wideval_20260703_0440.log`: resume step=30000,
  `lr: 0`, `[val] enabled: every 1 steps`):

```bash
cd ~/dev/glades-trainer && \
CHIRON_PACT_PROBE=1 CHIRON_PACT_PROBE_MAX=1 \
sh run.sh flagship --steps 30002 --accum 4 --lr 0 --warmup 0 --no-resume-warmup \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --whisc-ema 1.0 --val-every 1 --val-batches 1 <SAVE_SCRATCH_FLAGS> \
  2>&1 | tee logs/pact_m0_smoke_$(date +%Y%m%d_%H%M).log
```

Expected in the log: `[pact-m0] probe ENABLED maxSeq=1`, `[pact-m0] SELFTEST PASS`, resume from
`chiron_1B_pied_e4.final` at step 30000, then during the first val:
`[pact-m0] seq=1/1 gated_share=...% ungated_share=...% A_vs_p_relerr=...` followed by the
AGGREGATE line, chi_hist, and 24 `occ layer=` lines. The run then completes val normally.

- [ ] **Step 3: Validate the smoke.**
  - `A_vs_p_relerr` ≤ 0.1 (expected ~1e-2 or better). If > 0.1: STOP — investigate the D
    convention (try `Π_{l'>l}`, i.e. move the `suffix *= c` line above the `D[l] = suffix*c`
    assignment) or the hook placement (must be after the commit, before the next layer). Re-run.
  - No `[val]` NLL regression vs the wideval log's first-batch value (the probe must not perturb
    the forward — it only reads).
  - Flagship checkpoint mtime unchanged (compare against Step 1).

- [ ] **Step 4: Commit the log reference** (no code change — record the smoke in the arc notes;
  logs/ is typically untracked, do not force-add; nothing to commit if so).

---

### Task 4: Full M0 run (8 sequences), decision, and records

**Files:**
- Create: `~/dev/glades-ml/research/CHIRON_PACT_M0_2026_07_04.md`
- Modify: `~/dev/glades-ml/docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md`
  (tick the M0 line with the result)
- Produces: `~/dev/glades-trainer/logs/pact_m0_full_<ts>.log`

- [ ] **Step 1: Full probe run** — same invocation as Task 3 Step 2 with
  `CHIRON_PACT_PROBE_MAX=8` and `--val-batches 8`, tee to `logs/pact_m0_full_$(date +%Y%m%d_%H%M).log`.

Expected: 8 `seq=k/8` lines, then AGGREGATE + chi_hist + 24 occupancy lines. Runtime: minutes
(downloads ~6 GB/seq + host math ~30–60 s/seq).

- [ ] **Step 2: Extract the decision numbers.**

```bash
grep "pact-m0" ~/dev/glades-trainer/logs/pact_m0_full_*.log | grep -E "AGGREGATE|occ layer|chi_hist"
```

Decision (pre-registered, spec §13): **gated_share ≥ 3% ⇒ M0 GO** (proceed to the E0–E4
implementation plan); **< 3% ⇒ M0 KILL** (close the arc).

- [ ] **Step 3: Write the M0 record** to
  `~/dev/glades-ml/research/CHIRON_PACT_M0_2026_07_04.md`: setup (checkpoint, invocation, probe
  commit hashes), per-seq + aggregate gated/ungated shares, χ histogram, per-layer opposing-mass
  occupancy table, A-vs-p relerr (F2 validation + which D convention matched), the GO/KILL verdict
  against the 3% bar, and (if GO) the measured occupancy profile's implications for λ calibration
  at E2. House style: `research/CHIRON_PIED_E3_GATE_2026_07_02.md`.

- [ ] **Step 4: Update the spec's ladder** (§13 M0 bullet): append
  `**RESULT (2026-07-04): gated_share=X.XX% — GO/KILL.**`

- [ ] **Step 5: Commit both repos**

```bash
cd ~/dev/glades-ml && git add research/CHIRON_PACT_M0_2026_07_04.md \
  docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md && \
git commit -m "pact: M0 measurement gate result (<verdict>)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(If KILL: additionally revert the two probe commits in glades-trainer per the pq-probe precedent
and rebuild; if GO: keep the probe — it is env-gated, zero-cost when off, and doubles as the C3
occupancy observable for E3/E4.)

---

## Self-review

- **Spec coverage:** this plan implements exactly spec §13 M0 (probe, formulas §5.2, 3% kill bar,
  record). E0–E4 are explicitly out of scope, contingent on the M0 verdict — next plan.
- **Placeholder scan:** one deliberate parameterization: `<SAVE_SCRATCH_FLAGS>` is resolved by
  Task 3 Step 1 (grep run.sh) before use — the prior wideval artifact proves the mechanism exists.
- **Type consistency:** `g_pactProbe` fields match across Tasks 1–2 (`Mm` for the mass field to
  avoid shadowing math macros; `uCache`, `occNum/occDen`, `hostYpar/hostYperp/hostP` used
  identically in both tasks).
