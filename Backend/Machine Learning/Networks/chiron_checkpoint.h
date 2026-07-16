// chiron_checkpoint.h — CHRN/CHRF checkpoint format: single source of truth.
//
// Owns: the flag-bit registry, version rules, block codecs, the serving
// reader (model sections only), and model-section writer helpers.  The
// trainer (glades-trainer) composes its optimizer sections (Adam/Kahan/FACE)
// around these helpers; chiron_infer consumes the reader wholesale.
//
// Canonical CHRF section order (writer contract — reader depends on it):
//   magic 'CHRF' | version u32 | hdr i32[6]={T,m,L,nH,dH,V}
//   | step i32 | slcLastTransitionStep i32 | runtimeT i32 | runtimeL i32
//   | runtimeSasAlpha f32 | flags u32 | faceStepCount i32
//   | weights blob (E, then per layer Wq,Wk,Wv,Wo,gamma,beta[,gamma_p,beta_p])
//   | [bit 128] SCFA: k i32, w i32, per-layer D[m*(w+1)] f32
//   | [bit 256] QK-Norm gamma: L*nH f32
//   | [bits 2|16|32] Adam state | [bit 4] Kahan | [bit 1] FACE   (trainer-owned)
//   | [bit 8192] ORBIT optimizer state (trainer-owned; before all model tails)
//   | [bit 2048] WhiSC calibration: Pbar[L*m], Qbar[L*m], a[L*m] f32 (a used by serving)
//   | [bit 4096] causal-block SCFA marker (no payload; changes SCFA operator semantics)
//   | [bit 512] a_drift: L*m f32
//   | [bit 1024] rot_phi: L*m f32       <-- MUST STAY LAST (EOF-tail read)
// Version rules: v4 iff bit 64 (bf16-on-disk); else v3 iff bit 8 (gamma_p);
// else v2.  CHRN (legacy) v1..3; v3 prepends a u32 flags word (bits below).
//
// C++98.

#ifndef _GLADES_CHIRON_CHECKPOINT_H_
#define _GLADES_CHIRON_CHECKPOINT_H_

#include <cstdio>
#include <stdint.h>
#include <string>
#include <vector>
#include "cuda/gpu_buffer.h"

namespace glades {
namespace chiron {

// CHRF flags word bits (both binaries compile against this one registry).
enum ChironCkptBits
{
	CKPT_BIT_FACE         = 1,
	CKPT_BIT_BF16_ADAM    = 2,
	CKPT_BIT_KAHAN        = 4,
	CKPT_BIT_GAMMA_P      = 8,
	CKPT_BIT_INT8_ADAM    = 16,
	CKPT_BIT_FP32_ADAM    = 32,
	CKPT_BIT_BF16_DISK    = 64,
	CKPT_BIT_SCFA         = 128,
	CKPT_BIT_QKNORM_GAMMA = 256,
	CKPT_BIT_A_DRIFT      = 512,
	CKPT_BIT_ROT_PHI      = 1024,
	CKPT_BIT_WHISC_STATE  = 2048,
	CKPT_BIT_CAUSAL_SCFA  = 4096,
	CKPT_BIT_ORBIT_STATE  = 8192
};
// All bits the current format defines. WhiSC and ORBIT are trainer-owned and
// sit before the two EOF tails, so serving may safely ignore them after checking
// the bits. Unknown future bits still hard-error to protect tail arithmetic.
static const uint32_t CKPT_KNOWN_BITS_MASK = 0x3FFFu;  // == 16383 == bits 1..8192

// CHRN v3 legacy flags word bits.
enum ChironChrnBits
{
	CHRN_BIT_BF16_WEIGHTS = 1,
	CHRN_BIT_HAS_GAMMA_P  = 2
};

struct ChironModelDims
{
	int T, m, L, nH, dH, V, dModel;
	ChironModelDims() : T(0), m(0), L(0), nH(0), dH(0), V(0), dModel(0) {}
};

struct ChironScfaState
{
	int k;
	int w;
	bool present;      // SCFA forward active (resolve-time)
	bool dLoaded;      // D came from the checkpoint
	bool causalBlock;  // checkpoint declares exact-causal block SCFA semantics
	glades::gpu::GpuBuffer<float> B;                 // legacy DCT basis; unused by causal-block SCFA
	std::vector<glades::gpu::GpuBuffer<float>*> D;   // [L][m*(w+1)]
	ChironScfaState();
	~ChironScfaState();   // deletes D[i]
private:
	ChironScfaState(const ChironScfaState&);
	ChironScfaState& operator=(const ChironScfaState&);
};

struct ChironModelWeights
{
	glades::gpu::GpuBuffer<float> E;                       // [V,m]
	std::vector<glades::gpu::GpuBuffer<float>*> Wq, Wk, Wv, Wo;  // [m,dModel]/[dModel,m]
	std::vector<glades::gpu::GpuBuffer<float>*> gamma, beta;     // [m]
	std::vector<glades::gpu::GpuBuffer<float>*> gamma_p, beta_p; // [m]; empty if absent
	ChironScfaState scfa;
	std::vector<float> qknormGamma;  // L*nH iff bit 256, else empty
	std::vector<float> whiscA;       // L*m iff bit 2048; fixed causal serving calibration
	std::vector<float> rotPhi;       // L*m  iff bit 1024, else empty
	std::vector<float> aDrift;       // L*m  iff bit 512 (OBSD per-layer drift gate); loaded from EOF tail; empty if absent
	bool hasADrift;                  // bit 512 seen (aDrift is loaded; nonzero => refuse to serve)
	ChironModelWeights();
	~ChironModelWeights();  // deletes all per-layer buffers
private:
	ChironModelWeights(const ChironModelWeights&);
	ChironModelWeights& operator=(const ChironModelWeights&);
};

// ---- Block codecs (shared framing helpers) ----
// Read n weight elements (fp32, or bf16 widened to fp32 when bf16OnDisk).
// fp32buf is resized to n on success.
bool chiron_read_block(std::FILE* fp, std::vector<float>& fp32buf, size_t n, bool bf16OnDisk);
// Write n elements from fp32buf (as fp32, or RNE-rounded bf16).
bool chiron_write_block(std::FILE* fp, const std::vector<float>& fp32buf, size_t n, bool bf16OnDisk);

// ---- Optimizer-group codecs (ported from trainer chiron_main.cpp:5584-5740) ----
// uint16 bf16 group: u32 count sentinel + n uint16 values.
bool chiron_save_bf16_group_with_count(std::FILE* fp,
                                       const glades::gpu::GpuBuffer<uint16_t>* buf,
                                       size_t n);
bool chiron_load_bf16_group_with_count(std::FILE* fp,
                                       glades::gpu::GpuBuffer<uint16_t>* buf,
                                       size_t expected_n);
// int8 Adam group: u32 count_main, count_main*(int8+uint8), u32 count_scales, count_scales*(f32+f32).
bool chiron_save_int8_adam_group(std::FILE* fp,
                                 const glades::gpu::GpuBuffer<int8_t>*  mI,
                                 const glades::gpu::GpuBuffer<uint8_t>* vI,
                                 const glades::gpu::GpuBuffer<float>*   mS,
                                 const glades::gpu::GpuBuffer<float>*   vS,
                                 size_t n);
bool chiron_load_int8_adam_group(std::FILE* fp,
                                 glades::gpu::GpuBuffer<int8_t>*  mI,
                                 glades::gpu::GpuBuffer<uint8_t>* vI,
                                 glades::gpu::GpuBuffer<float>*   mS,
                                 glades::gpu::GpuBuffer<float>*   vS,
                                 size_t expected_n);
// fp32 Adam group: u32 count + count*(float m) + count*(float v).
bool chiron_save_fp32_adam_group(std::FILE* fp,
                                 const glades::gpu::GpuBuffer<float>* mF,
                                 const glades::gpu::GpuBuffer<float>* vF,
                                 size_t n);
bool chiron_load_fp32_adam_group(std::FILE* fp,
                                 glades::gpu::GpuBuffer<float>* mF,
                                 glades::gpu::GpuBuffer<float>* vF,
                                 size_t expected_n);

// ---- Header ----
struct ChironCkptHeader
{
	ChironModelDims dims;
	int32_t step;
	int32_t slcLastTransitionStep;
	int32_t runtimeT, runtimeL;
	float runtimeSasAlpha;
	uint32_t flags;
	int32_t faceStepCount;
	ChironCkptHeader() : step(0), slcLastTransitionStep(-1), runtimeT(0),
	                     runtimeL(0), runtimeSasAlpha(0.0f), flags(0), faceStepCount(0) {}
};
// Writes magic+version+hdr+meta+flags.  version derived from flags (see top).
bool chiron_write_header(std::FILE* fp, const ChironCkptHeader& h);

// ---- Model-section writers (byte layout owned here; data sourcing is caller's) ----
bool chiron_write_scfa_section(std::FILE* fp, int k, int w,
                               const std::vector<const float*>& D_host, size_t D_sz);
bool chiron_write_qknorm_section(std::FILE* fp, const float* gammaHost, size_t n); // n = L*nH
bool chiron_write_f32_tail_section(std::FILE* fp, const float* vals, size_t n);    // a_drift / rot_phi

// ---- Serving reader ----
// Loads model sections of a CHRN/CHRF checkpoint straight to GPU buffers.
// Returns true on success.  On failure: err gets a printable message and
// errCode gets 4 (load/parse failure) — callers map to their exit codes.
// Behavior is verbatim chiron_infer::loadCheckpoint (2026-07-03):
// sequential header/weights/SCFA/qknorm reads, rot_phi from the EOF tail,
// CHRN-v3 gamma_p file-size fallback, hard-error on flags & ~CKPT_KNOWN_BITS_MASK.
bool chiron_load_model(const std::string& path, ChironModelDims& dims,
                       ChironModelWeights& w, std::string& err, int& errCode);

} // namespace chiron
} // namespace glades

#endif
