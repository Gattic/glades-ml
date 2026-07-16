// chiron_checkpoint.cpp — CHRN/CHRF format module.
//
// Task 2: struct ctors/dtors.
// Task 3: block codecs + optimizer-group codecs ported from trainer.
// Task 4: header writer + model-section writers + serving reader.
//
// C++98.

#include "chiron_checkpoint.h"
#include "cuda/gpu_kernels.h"  // adam_int8_scale_count (glades::gpu)

#include <cstdio>
#include <cstring>
#include <vector>
#include <stdint.h>

namespace glades {
namespace chiron {

// ---------------------------------------------------------------------------
// ChironScfaState
// ---------------------------------------------------------------------------

ChironScfaState::ChironScfaState()
    : k(0), w(0), present(false), dLoaded(false), causalBlock(false)
{
}

ChironScfaState::~ChironScfaState()
{
    for (size_t i = 0; i < D.size(); ++i)
        delete D[i];
    D.clear();
}

// ---------------------------------------------------------------------------
// ChironModelWeights
// ---------------------------------------------------------------------------

ChironModelWeights::ChironModelWeights()
    : hasADrift(false)
{
}

ChironModelWeights::~ChironModelWeights()
{
    for (size_t i = 0; i < Wq.size();    ++i) delete Wq[i];
    for (size_t i = 0; i < Wk.size();    ++i) delete Wk[i];
    for (size_t i = 0; i < Wv.size();    ++i) delete Wv[i];
    for (size_t i = 0; i < Wo.size();    ++i) delete Wo[i];
    for (size_t i = 0; i < gamma.size(); ++i) delete gamma[i];
    for (size_t i = 0; i < beta.size();  ++i) delete beta[i];
    for (size_t i = 0; i < gamma_p.size(); ++i) delete gamma_p[i];
    for (size_t i = 0; i < beta_p.size(); ++i) delete beta_p[i];
    for (size_t i = 0; i < ffnGate.size(); ++i) delete ffnGate[i];
    for (size_t i = 0; i < ffnUp.size(); ++i) delete ffnUp[i];
    for (size_t i = 0; i < ffnDown.size(); ++i) delete ffnDown[i];
}

// ---------------------------------------------------------------------------
// BF16 host helpers (ported from trainer chiron_main.cpp:3191-3219)
// ---------------------------------------------------------------------------

// Round-to-nearest-even fp32 → bf16 encode.
// Matches the trainer's fp32_to_bf16_rne_host exactly (chiron_main.cpp:3191).
static inline uint16_t fp32_to_bf16_rne_host(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    if (f != f) // NaN
    {
        uint32_t sign = v.u & 0x80000000u;
        return (uint16_t)(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
    }
    uint32_t lsb  = (v.u >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    return (uint16_t)((v.u + bias) >> 16);
}

// bf16 u16 → fp32: left-shift into the top 16 bits.
// Matches the trainer's bf16_to_fp32_host (chiron_main.cpp:3214).
static inline float bf16_to_fp32_host(uint16_t b)
{
    union { float f; uint32_t u; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

// ---------------------------------------------------------------------------
// Block codecs (ported from trainer save_weight_block / load_weight_block,
// chiron_main.cpp:5741-5897).
// Changes vs trainer originals:
//   * chiron_read_block resizes fp32buf to n (trainer callers pre-sized it).
//   * chiron_write_block takes const& (no in-place mutation; local bf16 scratch).
//   * `static` dropped, `chiron_` prefix added.
// ---------------------------------------------------------------------------

// Read n weight elements from fp from disk into fp32buf.
// bf16OnDisk=true: each u16 is decoded via bf16_to_fp32_host.
// Corresponds to trainer's load_weight_block (chiron_main.cpp:5887).
bool chiron_read_block(std::FILE* fp, std::vector<float>& fp32buf,
                       size_t n, bool bf16OnDisk)
{
    fp32buf.resize(n);
    if (!bf16OnDisk)
        return std::fread(&fp32buf[0], sizeof(float), n, fp) == n;
    std::vector<uint16_t> bf16buf(n);
    if (std::fread(&bf16buf[0], sizeof(uint16_t), n, fp) != n) return false;
    for (size_t i = 0; i < n; ++i) fp32buf[i] = bf16_to_fp32_host(bf16buf[i]);
    return true;
}

// Write n elements from fp32buf to fp.
// bf16OnDisk=true: each element is RNE-rounded to u16 before writing.
// Corresponds to trainer's save_weight_block (chiron_main.cpp:5741).
bool chiron_write_block(std::FILE* fp, const std::vector<float>& fp32buf,
                        size_t n, bool bf16OnDisk)
{
    if (!bf16OnDisk)
        return std::fwrite(&fp32buf[0], sizeof(float), n, fp) == n;
    std::vector<uint16_t> bf16buf(n);
    for (size_t i = 0; i < n; ++i) bf16buf[i] = fp32_to_bf16_rne_host(fp32buf[i]);
    return std::fwrite(&bf16buf[0], sizeof(uint16_t), n, fp) == n;
}

// ---------------------------------------------------------------------------
// Optimizer-group codecs (ported from trainer chiron_main.cpp:5584-5727)
// Changes: `static` dropped, `chiron_` prefix added, log_error → fprintf(stderr).
// ---------------------------------------------------------------------------

// uint16 bf16 group: u32 count sentinel + n uint16 values.
// Corresponds to trainer's save_bf16_group_with_count (chiron_main.cpp:5584).
bool chiron_save_bf16_group_with_count(std::FILE* fp,
                                       const glades::gpu::GpuBuffer<uint16_t>* buf,
                                       size_t n)
{
    const uint32_t count = (buf == NULL || buf->allocated() == 0) ? 0u : (uint32_t)n;
    if (std::fwrite(&count, sizeof(uint32_t), 1, fp) != 1) return false;
    if (count == 0) return true;
    std::vector<uint16_t> tmp(n);
    buf->download(&tmp[0], n);
    return std::fwrite(&tmp[0], sizeof(uint16_t), n, fp) == n;
}

// Corresponds to trainer's load_bf16_group_with_count (chiron_main.cpp:5596).
bool chiron_load_bf16_group_with_count(std::FILE* fp,
                                       glades::gpu::GpuBuffer<uint16_t>* buf,
                                       size_t expected_n)
{
    uint32_t count = 0;
    if (std::fread(&count, sizeof(uint32_t), 1, fp) != 1) return false;
    if (count == 0)
        return true;
    if ((size_t)count != expected_n)
    {
        std::fprintf(stderr, "chiron: load bf16 group count mismatch (file=%u expected=%zu)\n",
                     count, expected_n);
        return false;
    }
    std::vector<uint16_t> tmp(count);
    if (std::fread(&tmp[0], sizeof(uint16_t), count, fp) != count) return false;
    if (buf == NULL || buf->allocated() == 0)
        return true;
    buf->upload(&tmp[0], count);
    return true;
}

// int8 Adam group.
// Corresponds to trainer's save_int8_adam_group (chiron_main.cpp:5630).
bool chiron_save_int8_adam_group(std::FILE* fp,
                                 const glades::gpu::GpuBuffer<int8_t>*  mI,
                                 const glades::gpu::GpuBuffer<uint8_t>* vI,
                                 const glades::gpu::GpuBuffer<float>*   mS,
                                 const glades::gpu::GpuBuffer<float>*   vS,
                                 size_t n)
{
    const uint32_t cMain = (mI == NULL || mI->allocated() == 0) ? 0u : (uint32_t)n;
    if (std::fwrite(&cMain, sizeof(uint32_t), 1, fp) != 1) return false;
    if (cMain == 0) return true;
    std::vector<int8_t>  mIh(n);
    std::vector<uint8_t> vIh(n);
    mI->download(&mIh[0], n);
    vI->download(&vIh[0], n);
    if (std::fwrite(&mIh[0], sizeof(int8_t),  n, fp) != n) return false;
    if (std::fwrite(&vIh[0], sizeof(uint8_t), n, fp) != n) return false;
    const size_t nScales = (size_t)glades::gpu::adam_int8_scale_count((int)n);
    const uint32_t cScales = (uint32_t)nScales;
    if (std::fwrite(&cScales, sizeof(uint32_t), 1, fp) != 1) return false;
    std::vector<float> mSh(nScales), vSh(nScales);
    mS->download(&mSh[0], nScales);
    vS->download(&vSh[0], nScales);
    if (std::fwrite(&mSh[0], sizeof(float), nScales, fp) != nScales) return false;
    if (std::fwrite(&vSh[0], sizeof(float), nScales, fp) != nScales) return false;
    return true;
}

// Corresponds to trainer's load_int8_adam_group (chiron_main.cpp:5657).
bool chiron_load_int8_adam_group(std::FILE* fp,
                                 glades::gpu::GpuBuffer<int8_t>*  mI,
                                 glades::gpu::GpuBuffer<uint8_t>* vI,
                                 glades::gpu::GpuBuffer<float>*   mS,
                                 glades::gpu::GpuBuffer<float>*   vS,
                                 size_t expected_n)
{
    uint32_t cMain = 0;
    if (std::fread(&cMain, sizeof(uint32_t), 1, fp) != 1) return false;
    if (cMain == 0) return true;
    if ((size_t)cMain != expected_n)
    {
        std::fprintf(stderr, "chiron: load int8 adam count mismatch (file=%u expected=%zu)\n",
                     cMain, expected_n);
        return false;
    }
    std::vector<int8_t>  mIh(expected_n);
    std::vector<uint8_t> vIh(expected_n);
    if (std::fread(&mIh[0], sizeof(int8_t),  expected_n, fp) != expected_n) return false;
    if (std::fread(&vIh[0], sizeof(uint8_t), expected_n, fp) != expected_n) return false;
    uint32_t cScales = 0;
    if (std::fread(&cScales, sizeof(uint32_t), 1, fp) != 1) return false;
    std::vector<float> mSh(cScales), vSh(cScales);
    if (std::fread(&mSh[0], sizeof(float), cScales, fp) != cScales) return false;
    if (std::fread(&vSh[0], sizeof(float), cScales, fp) != cScales) return false;
    if (mI != NULL && mI->allocated() > 0) mI->upload(&mIh[0], expected_n);
    if (vI != NULL && vI->allocated() > 0) vI->upload(&vIh[0], expected_n);
    if (mS != NULL && mS->allocated() > 0) mS->upload(&mSh[0], cScales);
    if (vS != NULL && vS->allocated() > 0) vS->upload(&vSh[0], cScales);
    return true;
}

// fp32 Adam group.
// Corresponds to trainer's save_fp32_adam_group (chiron_main.cpp:5692).
bool chiron_save_fp32_adam_group(std::FILE* fp,
                                 const glades::gpu::GpuBuffer<float>* mF,
                                 const glades::gpu::GpuBuffer<float>* vF,
                                 size_t n)
{
    const uint32_t count = (mF == NULL || mF->allocated() == 0) ? 0u : (uint32_t)n;
    if (std::fwrite(&count, sizeof(uint32_t), 1, fp) != 1) return false;
    if (count == 0) return true;
    std::vector<float> tmp(n);
    mF->download(&tmp[0], n);
    if (std::fwrite(&tmp[0], sizeof(float), n, fp) != n) return false;
    vF->download(&tmp[0], n);
    return std::fwrite(&tmp[0], sizeof(float), n, fp) == n;
}

// Corresponds to trainer's load_fp32_adam_group (chiron_main.cpp:5707).
bool chiron_load_fp32_adam_group(std::FILE* fp,
                                 glades::gpu::GpuBuffer<float>* mF,
                                 glades::gpu::GpuBuffer<float>* vF,
                                 size_t expected_n)
{
    uint32_t count = 0;
    if (std::fread(&count, sizeof(uint32_t), 1, fp) != 1) return false;
    if (count == 0) return true;
    if ((size_t)count != expected_n)
    {
        std::fprintf(stderr, "chiron: load fp32 adam count mismatch (file=%u expected=%zu)\n",
                     count, expected_n);
        return false;
    }
    std::vector<float> tmp(expected_n);
    if (std::fread(&tmp[0], sizeof(float), expected_n, fp) != expected_n) return false;
    if (mF != NULL && mF->allocated() > 0) mF->upload(&tmp[0], expected_n);
    if (std::fread(&tmp[0], sizeof(float), expected_n, fp) != expected_n) return false;
    if (vF != NULL && vF->allocated() > 0) vF->upload(&tmp[0], expected_n);
    return true;
}

// ---------------------------------------------------------------------------
// Header writer
// ---------------------------------------------------------------------------

// Writes CHRF magic + version (derived from flags) + hdr[6] + metadata + flags.
// Version rule (mirrors trainer save_full_checkpoint chiron_main.cpp:6059):
//   bit 64 (BF16_DISK) → v4; else bit 8 (GAMMA_P) → v3; else v2.
bool chiron_write_header(std::FILE* fp, const ChironCkptHeader& h)
{
    const char magic[4] = {'C','H','R','F'};
    const uint32_t version = (h.flags & (uint32_t)CKPT_BIT_BF16_DISK) ? 4u
                           : (h.flags & (uint32_t)CKPT_BIT_GAMMA_P)   ? 3u
                           :                                              2u;
    const int32_t hdr[6]      = { h.dims.T, h.dims.m, h.dims.L,
                                   h.dims.nH, h.dims.dH, h.dims.V };
    const int32_t step        = h.step;
    const int32_t slcLast     = h.slcLastTransitionStep;
    const int32_t runtimeT    = h.runtimeT;
    const int32_t runtimeL    = h.runtimeL;
    const float   runtimeAlpha= h.runtimeSasAlpha;
    const uint32_t flags      = h.flags;
    const int32_t faceStep    = h.faceStepCount;

    if (std::fwrite(magic,        1,              4, fp) != 4) return false;
    if (std::fwrite(&version,     sizeof(uint32_t), 1, fp) != 1) return false;
    if (std::fwrite(hdr,          sizeof(int32_t),  6, fp) != 6) return false;
    if (std::fwrite(&step,        sizeof(int32_t),  1, fp) != 1) return false;
    if (std::fwrite(&slcLast,     sizeof(int32_t),  1, fp) != 1) return false;
    if (std::fwrite(&runtimeT,    sizeof(int32_t),  1, fp) != 1) return false;
    if (std::fwrite(&runtimeL,    sizeof(int32_t),  1, fp) != 1) return false;
    if (std::fwrite(&runtimeAlpha,sizeof(float),    1, fp) != 1) return false;
    if (std::fwrite(&flags,       sizeof(uint32_t), 1, fp) != 1) return false;
    if (std::fwrite(&faceStep,    sizeof(int32_t),  1, fp) != 1) return false;
    return true;
}

bool chiron_write_architecture_section(std::FILE* fp, uint32_t flags,
                                       int nKVHeads, int ffnHidden)
{
    if ((flags & (uint32_t)CKPT_BIT_GQA) != 0)
    {
        const int32_t v = (int32_t)nKVHeads;
        if (std::fwrite(&v, sizeof(v), 1, fp) != 1) return false;
    }
    if ((flags & (uint32_t)CKPT_BIT_FFN) != 0)
    {
        const int32_t v = (int32_t)ffnHidden;
        if (std::fwrite(&v, sizeof(v), 1, fp) != 1) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Model-section writers
// ---------------------------------------------------------------------------

// SCFA section (bit 128): k i32, w i32, then L × D[m*(w+1)] FP32 rows.
// D_host must have D_host.size() == L pointers, each pointing to D_sz floats.
bool chiron_write_scfa_section(std::FILE* fp, int k, int w,
                               const std::vector<const float*>& D_host,
                               size_t D_sz)
{
    const int32_t scfaK = k;
    const int32_t scfaW = w;
    if (std::fwrite(&scfaK, sizeof(int32_t), 1, fp) != 1) return false;
    if (std::fwrite(&scfaW, sizeof(int32_t), 1, fp) != 1) return false;
    for (size_t l = 0; l < D_host.size(); ++l)
    {
        if (std::fwrite(D_host[l], sizeof(float), D_sz, fp) != D_sz) return false;
    }
    return true;
}

// QK-Norm gamma section (bit 256): n = L*nH FP32 values, flat row-major.
bool chiron_write_qknorm_section(std::FILE* fp, const float* gammaHost, size_t n)
{
    return std::fwrite(gammaHost, sizeof(float), n, fp) == n;
}

// Generic FP32 tail section — used for both a_drift (bit 512) and rot_phi (bit 1024).
// rot_phi MUST be written last (reader seeks from EOF).
bool chiron_write_f32_tail_section(std::FILE* fp, const float* vals, size_t n)
{
    return std::fwrite(vals, sizeof(float), n, fp) == n;
}

// ---------------------------------------------------------------------------
// Serving reader — port of chiron_infer::loadCheckpoint (2026-07-03).
//
// Mechanical transformations from the original:
//   * readBlock lambda → chiron_read_block(fp, buf, n, weightsBf16)
//   * fprintf(stderr,...)/return false → snprintf into err, errCode=4, fclose, return false
//   * printf("[chiron-infer] ...") → printf("[chiron-ckpt] ...") (text after tag identical)
//   * Bit constants → ChironCkptBits enum
//   * hasADrift set from CKPT_BIT_A_DRIFT (new — not in original reader)
//   * Output params → ChironModelWeights members
// ---------------------------------------------------------------------------

bool chiron_load_model(const std::string& path, ChironModelDims& dims,
                       ChironModelWeights& w, std::string& err, int& errCode)
{
#ifdef GLADES_HAVE_CUDA
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb), "chiron_infer: cannot open %s\n", path.c_str());
        err = eb; errCode = 4; return false;
    }

    char magic[4]; int version = 0; int hdr[6];
    if (std::fread(magic, 1, 4, fp) != 4)
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb), "chiron_infer: short magic in %s\n", path.c_str());
        err = eb; errCode = 4; std::fclose(fp); return false;
    }
    const bool isFull   = (std::memcmp(magic, "CHRF", 4) == 0);
    const bool isLegacy = (std::memcmp(magic, "CHRN", 4) == 0);
    if (!isFull && !isLegacy)
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb),
            "chiron_infer: bad magic in %s [%c%c%c%c]\n",
            path.c_str(), magic[0], magic[1], magic[2], magic[3]);
        err = eb; errCode = 4; std::fclose(fp); return false;
    }
    const int maxVer = isFull ? 4 : 3;
    const int minVer = isFull ? 2 : 1;
    if (std::fread(&version, sizeof(int), 1, fp) != 1
        || version < minVer || version > maxVer)
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb),
            "chiron_infer: bad version %d (expected %d-%d for %s)\n",
            version, minVer, maxVer, isFull ? "CHRF" : "CHRN");
        err = eb; errCode = 4; std::fclose(fp); return false;
    }
    // CHRN v3 prepends a flags word before the hdr; older CHRN versions don't.
    bool     weightsBf16           = false;
    uint32_t chrnFlags             = 0;
    bool     chrnFlagsHasGammaPBit = false;
    if (!isFull && version >= 3)
    {
        if (std::fread(&chrnFlags, sizeof(uint32_t), 1, fp) != 1)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: short CHRN flags\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        weightsBf16           = (chrnFlags & 0x1u) != 0;
        chrnFlagsHasGammaPBit = (chrnFlags & 0x2u) != 0;
    }

    if (std::fread(hdr, sizeof(int), 6, fp) != 6)
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb), "chiron_infer: short header\n");
        err = eb; errCode = 4; std::fclose(fp); return false;
    }
    // CHRF: 7×4-byte metadata block between dim-hdr and weights.
    // Skip 5 i32 (step, slcLast, runtimeT, runtimeL, runtimeAlpha), read flags u32, skip 1 i32.
    uint32_t chrfFlags = 0;
    int32_t nKVHeadsMeta = hdr[3];
    int32_t ffnHiddenMeta = 0;
    if (isFull)
    {
        std::fseek(fp, 5 * (int)sizeof(int32_t), SEEK_CUR);
        if (std::fread(&chrfFlags, sizeof(uint32_t), 1, fp) != 1)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: short CHRF flags\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        std::fseek(fp, 1 * (int)sizeof(int32_t), SEEK_CUR);
        weightsBf16 = (chrfFlags & 64u) != 0;
        std::printf("[chiron-ckpt] CHRF format detected (v=%d), flags=0x%x\n",
                    version, chrfFlags);
        // Hard-error on any bit above the known mask: a future section would corrupt
        // the rot_phi EOF-tail read (bit 1024 must stay last).
        if ((chrfFlags & ~CKPT_KNOWN_BITS_MASK) != 0)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb),
                "chiron_infer: unknown CHRF flags 0x%x — checkpoint format is newer than\n"
                "  this binary. The rot_phi EOF-tail read (bit 1024) may be wrong.\n"
                "  Rebuild chiron_infer against the matching trainer.\n", chrfFlags);
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        if ((chrfFlags & (uint32_t)CKPT_BIT_CAUSAL_SCFA) != 0
            && (chrfFlags & (uint32_t)CKPT_BIT_SCFA) == 0)
        {
            err = "chiron_infer: causal-block SCFA marker set without an SCFA section";
            errCode = 4; std::fclose(fp); return false;
        }
        if ((chrfFlags & (uint32_t)CKPT_BIT_GQA) != 0
            && std::fread(&nKVHeadsMeta, sizeof(int32_t), 1, fp) != 1)
        {
            err = "chiron_infer: short GQA architecture metadata";
            errCode = 4; std::fclose(fp); return false;
        }
        if ((chrfFlags & (uint32_t)CKPT_BIT_FFN) != 0
            && std::fread(&ffnHiddenMeta, sizeof(int32_t), 1, fp) != 1)
        {
            err = "chiron_infer: short FFN architecture metadata";
            errCode = 4; std::fclose(fp); return false;
        }
    }
    if (weightsBf16)
        std::printf("[chiron-ckpt] weights are bf16 on disk (2x smaller)\n");

    // Determine whether per-layer gamma_p/beta_p are present.
    bool hasPerLayerGammaP = isFull
        ? (version >= 3 && (chrfFlags & 8u) != 0)
        : (version >= 3 ? chrnFlagsHasGammaPBit : (version >= 2));
    // CHRN v3 file-size fallback for old checkpoints written before the bit was added.
    if (!isFull && version >= 3 && !chrnFlagsHasGammaPBit)
    {
        long curPos = std::ftell(fp);
        std::fseek(fp, 0, SEEK_END);
        long fileSize = std::ftell(fp);
        std::fseek(fp, curPos, SEEK_SET);
        const size_t elemSize = weightsBf16 ? 2 : 4;
        const long long Esz = (long long)hdr[5] * hdr[1] * (long long)elemSize;
        const long long perLayerBase = (long long)elemSize *
            (3LL * hdr[1] * hdr[3] * hdr[4]
             + (long long)hdr[3] * hdr[4] * hdr[1]
             + 2LL * hdr[1]);
        const long long perLayerGammaP = 2LL * (long long)elemSize * hdr[1];
        const long long sizeWithout = curPos + Esz + (long long)hdr[2] * perLayerBase;
        const long long sizeWith    = sizeWithout + (long long)hdr[2] * perLayerGammaP;
        if      (fileSize == sizeWith)    hasPerLayerGammaP = true;
        else if (fileSize == sizeWithout) hasPerLayerGammaP = false;
        else std::fprintf(stderr,
            "chiron_infer: CHRN v=3 file-size mismatch (%lld vs %lld w/ or %lld w/o gamma_p)"
            " — defaulting to %s\n",
            (long long)fileSize, sizeWith, sizeWithout,
            hasPerLayerGammaP ? "with" : "without");
    }

    dims.T      = hdr[0]; dims.m  = hdr[1]; dims.L  = hdr[2];
    dims.nH     = hdr[3]; dims.dH = hdr[4]; dims.V  = hdr[5];
    dims.nKVH   = nKVHeadsMeta; dims.ffnHidden = ffnHiddenMeta;
    dims.dModel = dims.nH * dims.dH;
    dims.dModelKV = dims.nKVH * dims.dH;
    if (dims.nKVH <= 0 || dims.nH % dims.nKVH != 0 || dims.ffnHidden < 0)
    {
        err = "chiron_infer: invalid GQA/FFN architecture metadata";
        errCode = 4; std::fclose(fp); return false;
    }

    std::printf("[chiron-ckpt] dims: T=%d m=%d L=%d nH=%d nKVH=%d dH=%d V=%d dModel=%d ffnH=%d\n",
                dims.T,dims.m,dims.L,dims.nH,dims.nKVH,dims.dH,dims.V,dims.dModel,dims.ffnHidden);

    // Load E [V, m].
    const size_t Esize = (size_t)dims.V * dims.m;
    std::vector<float> buf;
    if (!chiron_read_block(fp, buf, Esize, weightsBf16))
    {
        char eb[512];
        std::snprintf(eb, sizeof(eb), "chiron_infer: short read on E\n");
        err = eb; errCode = 4; std::fclose(fp); return false;
    }
    if (!w.E.allocate(Esize))
    {
        err = "chiron_infer: E allocate failed"; errCode = 4; std::fclose(fp); return false;
    }
    w.E.upload(&buf[0], Esize);

    // Load per-layer weights.
    const size_t Wq_size   = (size_t)dims.m * dims.dModel;
    const size_t Wkv_size  = (size_t)dims.m * dims.dModelKV;
    const size_t Wo_size   = (size_t)dims.dModel * dims.m;
    const size_t WffnUp_size = (size_t)dims.m * dims.ffnHidden;
    const size_t WffnDown_size = (size_t)dims.ffnHidden * dims.m;
    std::vector<float> gbuf;
    w.Wq.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    w.Wk.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    w.Wv.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    w.Wo.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    w.gamma.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    w.beta.assign(dims.L,  (glades::gpu::GpuBuffer<float>*)0);
    if (hasPerLayerGammaP)
    {
        w.gamma_p.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
        w.beta_p.assign(dims.L,  (glades::gpu::GpuBuffer<float>*)0);
    }
    else { w.gamma_p.clear(); w.beta_p.clear(); }
    if (dims.ffnHidden > 0)
    {
        w.ffnGate.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
        w.ffnUp.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
        w.ffnDown.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
    }

    for (int l = 0; l < dims.L; ++l)
    {
        w.Wq[l]    = new glades::gpu::GpuBuffer<float>(); w.Wq[l]->allocate(Wq_size);
        w.Wk[l]    = new glades::gpu::GpuBuffer<float>(); w.Wk[l]->allocate(Wkv_size);
        w.Wv[l]    = new glades::gpu::GpuBuffer<float>(); w.Wv[l]->allocate(Wkv_size);
        w.Wo[l]    = new glades::gpu::GpuBuffer<float>(); w.Wo[l]->allocate(Wo_size);
        w.gamma[l] = new glades::gpu::GpuBuffer<float>(); w.gamma[l]->allocate(dims.m);
        w.beta[l]  = new glades::gpu::GpuBuffer<float>(); w.beta[l]->allocate(dims.m);

        if (!chiron_read_block(fp, buf, Wq_size, weightsBf16))
        { err = "chiron_infer: short read (Wq)"; errCode = 4; std::fclose(fp); return false; }
        w.Wq[l]->upload(&buf[0], Wq_size);
        if (!chiron_read_block(fp, buf, Wkv_size, weightsBf16))
        { err = "chiron_infer: short read (Wk)"; errCode = 4; std::fclose(fp); return false; }
        w.Wk[l]->upload(&buf[0], Wkv_size);
        if (!chiron_read_block(fp, buf, Wkv_size, weightsBf16))
        { err = "chiron_infer: short read (Wv)"; errCode = 4; std::fclose(fp); return false; }
        w.Wv[l]->upload(&buf[0], Wkv_size);
        if (!chiron_read_block(fp, buf, Wo_size, weightsBf16))
        { err = "chiron_infer: short read (Wo)"; errCode = 4; std::fclose(fp); return false; }
        w.Wo[l]->upload(&buf[0], Wo_size);
        if (!chiron_read_block(fp, gbuf, (size_t)dims.m, weightsBf16))
        { err = "chiron_infer: short read (gamma)"; errCode = 4; std::fclose(fp); return false; }
        w.gamma[l]->upload(&gbuf[0], (size_t)dims.m);
        if (!chiron_read_block(fp, gbuf, (size_t)dims.m, weightsBf16))
        { err = "chiron_infer: short read (beta)"; errCode = 4; std::fclose(fp); return false; }
        w.beta[l]->upload(&gbuf[0], (size_t)dims.m);

        if (hasPerLayerGammaP)
        {
            w.gamma_p[l] = new glades::gpu::GpuBuffer<float>(); w.gamma_p[l]->allocate(dims.m);
            w.beta_p[l]  = new glades::gpu::GpuBuffer<float>(); w.beta_p[l]->allocate(dims.m);
            if (!chiron_read_block(fp, gbuf, (size_t)dims.m, weightsBf16))
            { err = "chiron_infer: short read (gamma_p)"; errCode = 4; std::fclose(fp); return false; }
            w.gamma_p[l]->upload(&gbuf[0], (size_t)dims.m);
            if (!chiron_read_block(fp, gbuf, (size_t)dims.m, weightsBf16))
            { err = "chiron_infer: short read (beta_p)"; errCode = 4; std::fclose(fp); return false; }
            w.beta_p[l]->upload(&gbuf[0], (size_t)dims.m);
        }
        if (dims.ffnHidden > 0)
        {
            w.ffnGate[l] = new glades::gpu::GpuBuffer<float>(); w.ffnGate[l]->allocate(WffnUp_size);
            w.ffnUp[l]   = new glades::gpu::GpuBuffer<float>(); w.ffnUp[l]->allocate(WffnUp_size);
            w.ffnDown[l] = new glades::gpu::GpuBuffer<float>(); w.ffnDown[l]->allocate(WffnDown_size);
            if (!chiron_read_block(fp, buf, WffnUp_size, weightsBf16))
            { err = "chiron_infer: short read (FFN gate)"; errCode = 4; std::fclose(fp); return false; }
            w.ffnGate[l]->upload(&buf[0], WffnUp_size);
            if (!chiron_read_block(fp, buf, WffnUp_size, weightsBf16))
            { err = "chiron_infer: short read (FFN up)"; errCode = 4; std::fclose(fp); return false; }
            w.ffnUp[l]->upload(&buf[0], WffnUp_size);
            if (!chiron_read_block(fp, buf, WffnDown_size, weightsBf16))
            { err = "chiron_infer: short read (FFN down)"; errCode = 4; std::fclose(fp); return false; }
            w.ffnDown[l]->upload(&buf[0], WffnDown_size);
        }
    }

    // Optional SCFA blob (CHRF bit 128).
    // Layout: scfa_k i32, scfa_w i32, then per layer D[m*(w+1)] FP32.
    if (isFull && (chrfFlags & (uint32_t)CKPT_BIT_SCFA) != 0)
    {
        int32_t scfaK = 0, scfaW = 0;
        if (std::fread(&scfaK, sizeof(int32_t), 1, fp) != 1
            || std::fread(&scfaW, sizeof(int32_t), 1, fp) != 1)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: short SCFA header\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        w.scfa.k = scfaK;
        w.scfa.w = scfaW;
        const size_t D_sz = (size_t)dims.m * (size_t)(scfaW + 1);
        w.scfa.D.assign(dims.L, (glades::gpu::GpuBuffer<float>*)0);
        std::vector<float> Dh(D_sz);
        for (int l = 0; l < dims.L; ++l)
        {
            if (std::fread(&Dh[0], sizeof(float), D_sz, fp) != D_sz)
            {
                char eb[512];
                std::snprintf(eb, sizeof(eb), "chiron_infer: short SCFA D[%d]\n", l);
                err = eb; errCode = 4; std::fclose(fp); return false;
            }
            w.scfa.D[l] = new glades::gpu::GpuBuffer<float>();
            w.scfa.D[l]->allocate(D_sz);
            w.scfa.D[l]->upload(&Dh[0], D_sz);
        }
        w.scfa.present = true;
        w.scfa.dLoaded = true;
        w.scfa.causalBlock = (chrfFlags & (uint32_t)CKPT_BIT_CAUSAL_SCFA) != 0;
        std::printf("[chiron-ckpt] loaded %s SCFA state (k=%d w=%d, %.1f MB)\n",
                    w.scfa.causalBlock ? "causal-block" : "LEGACY NONCAUSAL",
                    scfaK, scfaW,
                    (double)((size_t)dims.L * D_sz * sizeof(float)) / (1024.0 * 1024.0));
    }

    // Optional per-head QK-Norm gamma (CHRF bit 256), L*nH FP32.
    if (isFull && (chrfFlags & (uint32_t)CKPT_BIT_QKNORM_GAMMA) != 0)
    {
        w.qknormGamma.assign((size_t)dims.L * dims.nH, 0.0f);
        if (std::fread(&w.qknormGamma[0], sizeof(float), w.qknormGamma.size(), fp)
                != w.qknormGamma.size())
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: short QK-Norm gamma blob\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        std::printf("[chiron-ckpt] loaded per-head QK-Norm gamma (L=%d nH=%d)"
                    " -- exact (not gamma~14 approx)\n",
                    dims.L, dims.nH);
    }

    // bit 2048: load the cached WhiSC `a` from the trainer-owned Pbar/Qbar/a
    // section. Serving treats it as fixed model state; recalibrating from the
    // current full window would let future tokens change earlier logits.
    if (isFull && (chrfFlags & (uint32_t)CKPT_BIT_WHISC_STATE) != 0)
    {
        const size_t nWhisc = (size_t)dims.L * dims.m;
        const long secBytes = (long)(nWhisc * sizeof(float));
        const int tailsAfterA = 1
            + ((chrfFlags & (uint32_t)CKPT_BIT_A_DRIFT) != 0 ? 1 : 0)
            + ((chrfFlags & (uint32_t)CKPT_BIT_ROT_PHI) != 0 ? 1 : 0);
        if (std::fseek(fp, 0, SEEK_END) != 0)
        {
            err = "chiron_infer: seek(END) failed for WhiSC a";
            errCode = 4; std::fclose(fp); return false;
        }
        const long fileSize = std::ftell(fp);
        const long need = (long)tailsAfterA * secBytes;
        if (fileSize < need || std::fseek(fp, fileSize - need, SEEK_SET) != 0)
        {
            err = "chiron_infer: invalid WhiSC a tail offset";
            errCode = 4; std::fclose(fp); return false;
        }
        w.whiscA.assign(nWhisc, 1.0f);
        if (std::fread(&w.whiscA[0], sizeof(float), nWhisc, fp) != nWhisc)
        {
            err = "chiron_infer: short read on WhiSC a (bit 2048)";
            errCode = 4; std::fclose(fp); return false;
        }
        std::printf("[chiron-ckpt] loaded fixed causal WhiSC a (L=%d m=%d, bit 2048)\n",
                    dims.L, dims.m);
    }

    // bit 512: OBSD per-layer drift gate a_drift (L*m FP32) — read from EOF tail.
    // a_drift is written immediately BEFORE rot_phi (bit 1024, which is always LAST).
    // Locate it by EOF arithmetic, mirroring the rot_phi tail read below:
    //   offset = fileSize - (rot_phi present ? 2 : 1) * L*m*4
    // Loading the values lets the serving resolver distinguish a genuinely
    // drift-TRAINED checkpoint (nonzero -> refuse) from the zero-init a_drift that
    // --whisc-coupling co-allocates (all-zero -> exact no-op, serve normally).
    w.hasADrift = isFull && (chrfFlags & (uint32_t)CKPT_BIT_A_DRIFT) != 0;
    if (w.hasADrift)
    {
        const size_t nDrift     = (size_t)dims.L * dims.m;
        const bool   rotPresent = (chrfFlags & (uint32_t)CKPT_BIT_ROT_PHI) != 0;
        w.aDrift.assign(nDrift, 0.0f);
        if (std::fseek(fp, 0, SEEK_END) != 0)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: seek(END) failed for a_drift\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        const long fileSize = std::ftell(fp);
        const long secBytes = (long)(nDrift * sizeof(float));
        const long need     = (rotPresent ? 2 : 1) * secBytes;  // bytes from a_drift start to EOF
        if (fileSize < need)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb),
                "chiron_infer: file too small for a_drift (%ld < %ld)\n", fileSize, need);
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        if (std::fseek(fp, fileSize - need, SEEK_SET) != 0)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: seek to a_drift tail failed\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        if (std::fread(&w.aDrift[0], sizeof(float), nDrift, fp) != nDrift)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb),
                "chiron_infer: short read on a_drift (bit 512)\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        std::printf("[chiron-ckpt] loaded OBSD a_drift gate (L=%d m=%d, bit 512)"
                    " -- read from EOF tail\n", dims.L, dims.m);
    }

    // Optional WhiSC-D rot_phi (CHRF bit 1024), L*m FP32 — read from EOF tail.
    // save_full always writes this section LAST; we seek from EOF to skip all
    // intermediate sections (Adam / Kahan / FACE / a_drift) without parsing them.
    if (isFull && (chrfFlags & (uint32_t)CKPT_BIT_ROT_PHI) != 0)
    {
        const size_t nRot = (size_t)dims.L * dims.m;
        w.rotPhi.assign(nRot, 0.0f);
        if (std::fseek(fp, 0, SEEK_END) != 0)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: seek(END) failed for rot_phi\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        const long fileSize = std::ftell(fp);
        const long need     = (long)(nRot * sizeof(float));
        if (fileSize < need)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb),
                "chiron_infer: file too small for rot_phi (%ld < %ld)\n", fileSize, need);
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        if (std::fseek(fp, fileSize - need, SEEK_SET) != 0)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb), "chiron_infer: seek to rot_phi tail failed\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        if (std::fread(&w.rotPhi[0], sizeof(float), nRot, fp) != nRot)
        {
            char eb[512];
            std::snprintf(eb, sizeof(eb),
                "chiron_infer: short read on rot_phi (bit 1024)\n");
            err = eb; errCode = 4; std::fclose(fp); return false;
        }
        std::printf("[chiron-ckpt] loaded WhiSC-D rot_phi angle (L=%d m=%d, bit 1024)"
                    " -- read from EOF tail\n", dims.L, dims.m);
    }

    std::fclose(fp);
    std::printf("[chiron-ckpt] loaded %d layers + embedding (%.1f MB)\n",
                dims.L,
                (double)((size_t)Esize + (size_t)dims.L
                          * (Wq_size + 2u*Wkv_size + Wo_size + 2u*(size_t)dims.m
                             + 2u*WffnUp_size + WffnDown_size))
                * 4.0 / (1024.0 * 1024.0));
    return true;
#else
    (void)path; (void)dims; (void)w;
    err     = "CUDA required";
    errCode = 4;
    return false;
#endif
}

} // namespace chiron
} // namespace glades
