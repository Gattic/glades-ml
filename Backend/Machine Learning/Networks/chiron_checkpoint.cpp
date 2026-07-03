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
    : k(0), w(0), present(false), dLoaded(false)
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
// Header writer stub
// ---------------------------------------------------------------------------

bool chiron_write_header(std::FILE* /*fp*/, const ChironCkptHeader& /*h*/)
{
    return false;
}

// ---------------------------------------------------------------------------
// Model-section writer stubs
// ---------------------------------------------------------------------------

bool chiron_write_scfa_section(std::FILE* /*fp*/, int /*k*/, int /*w*/,
                               const std::vector<const float*>& /*D_host*/,
                               size_t /*D_sz*/)
{
    return false;
}

bool chiron_write_qknorm_section(std::FILE* /*fp*/, const float* /*gammaHost*/,
                                 size_t /*n*/)
{
    return false;
}

bool chiron_write_f32_tail_section(std::FILE* /*fp*/, const float* /*vals*/,
                                   size_t /*n*/)
{
    return false;
}

// ---------------------------------------------------------------------------
// Serving reader stub
// ---------------------------------------------------------------------------

bool chiron_load_model(const std::string& /*path*/, ChironModelDims& /*dims*/,
                       ChironModelWeights& /*w*/, std::string& err, int& errCode)
{
    err     = "unimplemented";
    errCode = 4;
    return false;
}

} // namespace chiron
} // namespace glades
