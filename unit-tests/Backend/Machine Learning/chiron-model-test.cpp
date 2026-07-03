// chiron-model-test.cpp — CHIRON model checkpoint codec unit tests.
//
// Tests chiron_read_block / chiron_write_block (pure-CPU tmpfile roundtrips).
// The optimizer-group codecs (bf16/int8/fp32 Adam) are host+GPU and are
// exercised end-to-end in Task 12; only the block codecs are covered here.
//
// bf16 encode note: the trainer uses fp32_to_bf16_rne_host (round-to-nearest-
// even), NOT simple truncation (v.u >> 16).  The expected values below are
// computed with the same RNE algorithm so the test matches the encoder exactly.
//
// C++98.

#include "chiron-model-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/chiron_checkpoint.h"
#include "../../../Backend/Machine Learning/Networks/chiron_serving.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_chiron.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_blas.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <stdint.h>
#include <unistd.h>  // mkstemp, close, unlink

// ---------------------------------------------------------------------------
// Local RNE bf16 helpers — mirror the trainer's fp32_to_bf16_rne_host /
// bf16_to_fp32_host for computing expected values.  Must stay in sync with
// chiron_checkpoint.cpp's encode.
// ---------------------------------------------------------------------------

static uint16_t ut_fp32_to_bf16_rne(float f)
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

static float ut_bf16_to_fp32(uint16_t b)
{
	union { float f; uint32_t u; } v;
	v.u = ((uint32_t)b) << 16;
	return v.f;
}

// ---------------------------------------------------------------------------
// Test 1: fp32/bf16 block roundtrip via tmpfile().
// ---------------------------------------------------------------------------

void CHIRONCkptBlockCodecTest()
{
	std::vector<float> src(37);
	for (size_t i = 0; i < src.size(); ++i) src[i] = 0.5f * (float)i - 3.0f;

	// --- fp32: exact roundtrip ---
	std::FILE* fp = std::tmpfile();
	ASSERT("tmpfile", fp != 0);
	ASSERT("write fp32", glades::chiron::chiron_write_block(fp, src, src.size(), false));
	std::rewind(fp);
	std::vector<float> got;
	ASSERT("read fp32", glades::chiron::chiron_read_block(fp, got, src.size(), false));
	for (size_t i = 0; i < src.size(); ++i)
		ASSERT("fp32 exact", got[i] == src[i]);
	std::fclose(fp);

	// --- bf16: roundtrip via RNE encode (matches trainer fp32_to_bf16_rne_host) ---
	// NOTE: brief assumed truncation (v.u &= 0xFFFF0000u); trainer actually uses
	// round-to-nearest-even (chiron_main.cpp:5748).  Test uses ut_fp32_to_bf16_rne.
	fp = std::tmpfile();
	ASSERT("write bf16", glades::chiron::chiron_write_block(fp, src, src.size(), true));
	std::rewind(fp);
	ASSERT("read bf16", glades::chiron::chiron_read_block(fp, got, src.size(), true));
	for (size_t i = 0; i < src.size(); ++i)
	{
		float expected = ut_bf16_to_fp32(ut_fp32_to_bf16_rne(src[i]));
		ASSERT("bf16 roundtrip", got[i] == expected);
	}
	std::fclose(fp);
}

// ---------------------------------------------------------------------------
// Test 2: bf16 RNE-discriminating cases (review fix).
//
// The inputs in CHIRONCkptBlockCodecTest (0.5f*i-3.0f) are all exactly
// representable in bf16, so RNE and plain truncation agree for every one.
// This test exercises inputs whose low 16 bits are non-zero, where the two
// algorithms diverge — proving the encoder implements RNE, not truncate.
//
// Algorithm (from chiron_checkpoint.cpp / trainer chiron_main.cpp:3191):
//   lsb  = (u >> 16) & 1
//   bias = 0x7FFF + lsb
//   result = (u + bias) >> 16
//
// Hand derivation for each case (verify before changing expected values):
//
//   Case A: u=0x3F808001  low16=0x8001 > 0x8000  lsb=0  bias=0x7FFF
//     RNE:  (0x3F808001+0x7FFF)>>16 = 0x3F810000>>16 = 0x3F81
//     Trunc: 0x3F808001>>16 = 0x3F80        ← RNE != Trunc (rounds UP)
//
//   Case B: u=0x3F818000  low16=0x8000 (tie)  lsb=1(odd)  bias=0x8000
//     RNE:  (0x3F818000+0x8000)>>16 = 0x3F820000>>16 = 0x3F82
//     Trunc: 0x3F818000>>16 = 0x3F81        ← RNE != Trunc (round-up-to-even)
//
//   Case C: u=0x3F808000  low16=0x8000 (tie)  lsb=0(even)  bias=0x7FFF
//     RNE:  (0x3F808000+0x7FFF)>>16 = 0x3F80FFFF>>16 = 0x3F80
//     Trunc: 0x3F808000>>16 = 0x3F80        RNE == Trunc (round-down-stays)
//     (tests the stay-case so the tie coverage is complete)
// ---------------------------------------------------------------------------

static void CHIRONCkptBf16RneDiscriminatingTest()
{
	// Build three floats from explicit u32 bit patterns via union.
	union { float f; uint32_t u; } a, b, c;
	a.u = 0x3F808001u; // Case A: low16 > 0x8000 → RNE rounds UP
	b.u = 0x3F818000u; // Case B: tie, bit16 odd  → RNE rounds UP to even
	c.u = 0x3F808000u; // Case C: tie, bit16 even → RNE rounds DOWN (stays)

	// Hard-coded expected u16 bit patterns (derived above, NOT from ut_fp32_to_bf16_rne).
	const uint16_t expected_a = 0x3F81u; // RNE; truncation would give 0x3F80
	const uint16_t expected_b = 0x3F82u; // RNE; truncation would give 0x3F81
	const uint16_t expected_c = 0x3F80u; // RNE == truncation for even-tie case

	std::vector<float> src(3);
	src[0] = a.f;
	src[1] = b.f;
	src[2] = c.f;

	std::FILE* fp = std::tmpfile();
	ASSERT("rne_disc tmpfile", fp != 0);
	ASSERT("rne_disc write", glades::chiron::chiron_write_block(fp, src, src.size(), true));

	// Read raw u16 bytes and compare to hard-coded expected bit patterns.
	// This directly tests the encoded bits — not just the decoded float.
	std::rewind(fp);
	uint16_t raw[3] = {0, 0, 0};
	size_t nread = std::fread(raw, sizeof(uint16_t), 3, fp);
	ASSERT("rne_disc fread count", nread == 3);
	ASSERT("rne_disc case_a bits", raw[0] == expected_a);
	ASSERT("rne_disc case_b bits", raw[1] == expected_b);
	ASSERT("rne_disc case_c bits", raw[2] == expected_c);

	// Also verify the decoded-float path for completeness.
	std::rewind(fp);
	std::vector<float> got;
	ASSERT("rne_disc read_block", glades::chiron::chiron_read_block(fp, got, src.size(), true));
	ASSERT("rne_disc case_a float", got[0] == ut_bf16_to_fp32(expected_a));
	ASSERT("rne_disc case_b float", got[1] == ut_bf16_to_fp32(expected_b));
	ASSERT("rne_disc case_c float", got[2] == ut_bf16_to_fp32(expected_c));

	std::fclose(fp);
}

// ---------------------------------------------------------------------------
// Roundtrip test helpers (Task 4).
//
// Tiny shape: T=8, m=4, L=2, nH=1, dH=4, V=16, dModel=4.
// ---------------------------------------------------------------------------

static const int RT_T  = 8;
static const int RT_M  = 4;
static const int RT_L  = 2;
static const int RT_NH = 1;
static const int RT_DH = 4;
static const int RT_V  = 16;
static const int RT_DM = 4;  // nH * dH

// Create a temp file path via mkstemp (closes the fd; caller opens via fopen).
static std::string rt_tmppath()
{
	char path[] = "/tmp/chiron_rt_XXXXXX";
	int fd = mkstemp(path);
	if (fd < 0) return std::string();
	close(fd);
	return std::string(path);
}

// Fill n floats with sin(0.1*i + off).
static void rt_fill(float* v, size_t n, float off)
{
	for (size_t i = 0; i < n; ++i)
		v[i] = std::sin(0.1f * (float)i + off);
}

// Expected value after bf16 roundtrip (or identity for fp32).
static float rt_expect(float f, bool bf16)
{
	return bf16 ? ut_bf16_to_fp32(ut_fp32_to_bf16_rne(f)) : f;
}

// Download buf and compare element-by-element against src[0..n-1].
// ASSERTS both the download return and every element.
static void rt_cmpbuf(const char* tag,
                      const glades::gpu::GpuBuffer<float>& buf,
                      const float* src, size_t n, bool bf16)
{
	std::vector<float> got(n);
	ASSERT(tag, buf.download(&got[0], n));
	for (size_t i = 0; i < n; ++i)
		ASSERT(tag, got[i] == rt_expect(src[i], bf16));
}

// Per-layer weight data for the roundtrip tests.
struct RTLayerWeights
{
	std::vector<float> Wq, Wk, Wv, Wo;   // [m*dModel], [m*dModel], [m*dModel], [dModel*m]
	std::vector<float> gamma, beta;        // [m]
	std::vector<float> gamma_p, beta_p;   // [m] — only filled when hasGammaP
};

struct RTWeightData
{
	std::vector<float> E;           // [V*m]
	RTLayerWeights     layers[2];   // RT_L == 2
};

// Fill d with deterministic values; gamma_p/beta_p filled when hasGammaP.
static void rt_fill_weights(RTWeightData& d, bool hasGammaP)
{
	const size_t Esize   = (size_t)RT_V * RT_M;
	const size_t Wqkv_sz = (size_t)RT_M * RT_DM;
	const size_t Wo_sz   = (size_t)RT_DM * RT_M;

	d.E.resize(Esize);
	rt_fill(&d.E[0], Esize, 0.0f);

	for (int l = 0; l < RT_L; ++l)
	{
		float off = 1.0f + (float)l * 9.0f;
		d.layers[l].Wq.resize(Wqkv_sz); rt_fill(&d.layers[l].Wq[0], Wqkv_sz, off + 0);
		d.layers[l].Wk.resize(Wqkv_sz); rt_fill(&d.layers[l].Wk[0], Wqkv_sz, off + 1);
		d.layers[l].Wv.resize(Wqkv_sz); rt_fill(&d.layers[l].Wv[0], Wqkv_sz, off + 2);
		d.layers[l].Wo.resize(Wo_sz);   rt_fill(&d.layers[l].Wo[0],   Wo_sz, off + 3);
		d.layers[l].gamma.resize(RT_M); rt_fill(&d.layers[l].gamma[0], RT_M, off + 4);
		d.layers[l].beta.resize(RT_M);  rt_fill(&d.layers[l].beta[0],  RT_M, off + 5);
		if (hasGammaP)
		{
			d.layers[l].gamma_p.resize(RT_M); rt_fill(&d.layers[l].gamma_p[0], RT_M, off + 6);
			d.layers[l].beta_p.resize(RT_M);  rt_fill(&d.layers[l].beta_p[0],  RT_M, off + 7);
		}
	}
}

// Write weights blob (E + per-layer) to fp using chiron_write_block.
static bool rt_write_weights(std::FILE* fp, const RTWeightData& d, bool bf16, bool hasGammaP)
{
	const size_t Esize   = (size_t)RT_V * RT_M;
	const size_t Wqkv_sz = (size_t)RT_M * RT_DM;
	const size_t Wo_sz   = (size_t)RT_DM * RT_M;

	if (!glades::chiron::chiron_write_block(fp, d.E, Esize, bf16)) return false;
	for (int l = 0; l < RT_L; ++l)
	{
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].Wq,    Wqkv_sz, bf16)) return false;
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].Wk,    Wqkv_sz, bf16)) return false;
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].Wv,    Wqkv_sz, bf16)) return false;
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].Wo,    Wo_sz,   bf16)) return false;
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].gamma, RT_M,    bf16)) return false;
		if (!glades::chiron::chiron_write_block(fp, d.layers[l].beta,  RT_M,    bf16)) return false;
		if (hasGammaP)
		{
			if (!glades::chiron::chiron_write_block(fp, d.layers[l].gamma_p, RT_M, bf16)) return false;
			if (!glades::chiron::chiron_write_block(fp, d.layers[l].beta_p,  RT_M, bf16)) return false;
		}
	}
	return true;
}

// Verify loaded ChironModelWeights against the source RTWeightData.
static void rt_verify_weights(const RTWeightData& src,
                              const glades::chiron::ChironModelWeights& w,
                              bool bf16, bool hasGammaP)
{
	const size_t Esize   = (size_t)RT_V * RT_M;
	const size_t Wqkv_sz = (size_t)RT_M * RT_DM;
	const size_t Wo_sz   = (size_t)RT_DM * RT_M;

	rt_cmpbuf("rt E",    w.E,         &src.E[0], Esize, bf16);
	for (int l = 0; l < RT_L; ++l)
	{
		rt_cmpbuf("rt Wq",    *w.Wq[l],    &src.layers[l].Wq[0],    Wqkv_sz, bf16);
		rt_cmpbuf("rt Wk",    *w.Wk[l],    &src.layers[l].Wk[0],    Wqkv_sz, bf16);
		rt_cmpbuf("rt Wv",    *w.Wv[l],    &src.layers[l].Wv[0],    Wqkv_sz, bf16);
		rt_cmpbuf("rt Wo",    *w.Wo[l],    &src.layers[l].Wo[0],    Wo_sz,   bf16);
		rt_cmpbuf("rt gamma", *w.gamma[l], &src.layers[l].gamma[0], RT_M,    bf16);
		rt_cmpbuf("rt beta",  *w.beta[l],  &src.layers[l].beta[0],  RT_M,    bf16);
		if (hasGammaP)
		{
			rt_cmpbuf("rt gamma_p", *w.gamma_p[l], &src.layers[l].gamma_p[0], RT_M, bf16);
			rt_cmpbuf("rt beta_p",  *w.beta_p[l],  &src.layers[l].beta_p[0],  RT_M, bf16);
		}
	}
}

// Build a ChironCkptHeader with the tiny test shape.
static glades::chiron::ChironCkptHeader rt_make_hdr(uint32_t flags)
{
	glades::chiron::ChironCkptHeader h;
	h.dims.T      = RT_T;
	h.dims.m      = RT_M;
	h.dims.L      = RT_L;
	h.dims.nH     = RT_NH;
	h.dims.dH     = RT_DH;
	h.dims.V      = RT_V;
	h.dims.dModel = RT_DM;
	h.step                  = 100;
	h.slcLastTransitionStep = -1;
	h.runtimeT              = RT_T;
	h.runtimeL              = RT_L;
	h.runtimeSasAlpha       = 0.0f;
	h.flags                 = flags;
	h.faceStepCount         = 0;
	return h;
}

// Test-local CHRN v1 writer (weights only, no gamma_p).
// Layout: magic[4] version[4] hdr[6*4] weights-blob.
static bool rt_write_chrn_v1(std::FILE* fp, const RTWeightData& d)
{
	const char magic[4] = {'C','H','R','N'};
	const uint32_t version = 1u;
	const int32_t hdr[6] = { RT_T, RT_M, RT_L, RT_NH, RT_DH, RT_V };
	if (std::fwrite(magic,    1,             4, fp) != 4) return false;
	if (std::fwrite(&version, sizeof(uint32_t), 1, fp) != 1) return false;
	if (std::fwrite(hdr,      sizeof(int32_t),  6, fp) != 6) return false;
	return rt_write_weights(fp, d, false, false);
}

// Test-local CHRN v3 writer.
// Layout: magic[4] version[4] chrnFlags[4] hdr[6*4] weights-blob.
static bool rt_write_chrn_v3(std::FILE* fp, const RTWeightData& d, uint32_t chrnFlags)
{
	const char magic[4] = {'C','H','R','N'};
	const uint32_t version = 3u;
	const int32_t hdr[6] = { RT_T, RT_M, RT_L, RT_NH, RT_DH, RT_V };
	if (std::fwrite(magic,      1,             4, fp) != 4) return false;
	if (std::fwrite(&version,   sizeof(uint32_t), 1, fp) != 1) return false;
	if (std::fwrite(&chrnFlags, sizeof(uint32_t), 1, fp) != 1) return false;
	if (std::fwrite(hdr,        sizeof(int32_t),  6, fp) != 6) return false;
	bool bf16      = (chrnFlags & (uint32_t)glades::chiron::CHRN_BIT_BF16_WEIGHTS) != 0;
	bool hasGammaP = (chrnFlags & (uint32_t)glades::chiron::CHRN_BIT_HAS_GAMMA_P)  != 0;
	return rt_write_weights(fp, d, bf16, hasGammaP);
}

// ---------------------------------------------------------------------------
// Test 3: Full reader + writer roundtrip (cases 1-7, incl. 6b a_drift-last).
// ---------------------------------------------------------------------------

static void CHIRONCkptRoundtripTest()
{
	// ---- Case 1: Minimal CHRF v2 (flags=0), weights only ----
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		std::string path = rt_tmppath();
		ASSERT("rt1 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt1 fopen", fp != 0);
		ASSERT("rt1 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr(0)));
		ASSERT("rt1 write_weights", rt_write_weights(fp, src, false, false));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt1 load",    glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		ASSERT("rt1 T",       dims.T      == RT_T);
		ASSERT("rt1 m",       dims.m      == RT_M);
		ASSERT("rt1 L",       dims.L      == RT_L);
		ASSERT("rt1 dModel",  dims.dModel == RT_DM);
		rt_verify_weights(src, w, false, false);
		ASSERT("rt1 no gamma_p", w.gamma_p.empty());
		ASSERT("rt1 no scfa",    !w.scfa.dLoaded);
		ASSERT("rt1 no qknorm",  w.qknormGamma.empty());
		ASSERT("rt1 no rotphi",  w.rotPhi.empty());
		ASSERT("rt1 no adrift",  !w.hasADrift);

		unlink(path.c_str());
	}

	// ---- Case 2: CHRF v3 with gamma_p (bit 8) ----
	{
		RTWeightData src;
		rt_fill_weights(src, true);

		std::string path = rt_tmppath();
		ASSERT("rt2 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt2 fopen", fp != 0);
		ASSERT("rt2 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr((uint32_t)glades::chiron::CKPT_BIT_GAMMA_P)));
		ASSERT("rt2 write_weights", rt_write_weights(fp, src, false, true));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt2 load",           glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		rt_verify_weights(src, w, false, true);
		ASSERT("rt2 gamma_p present", (int)w.gamma_p.size() == RT_L);
		ASSERT("rt2 beta_p present",  (int)w.beta_p.size()  == RT_L);

		unlink(path.c_str());
	}

	// ---- Case 3: CHRF v4 bf16-on-disk (bit 64) ----
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		std::string path = rt_tmppath();
		ASSERT("rt3 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt3 fopen", fp != 0);
		ASSERT("rt3 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr((uint32_t)glades::chiron::CKPT_BIT_BF16_DISK)));
		ASSERT("rt3 write_weights", rt_write_weights(fp, src, true, false));  // bf16 on disk
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt3 load", glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		rt_verify_weights(src, w, true, false);  // compare with bf16-truncated source

		unlink(path.c_str());
	}

	// ---- Case 4: SCFA(128) + qknorm(256) + rot_phi(1024) + dummy fp32-Adam(32) ----
	// The dummy 100-float optimizer payload (bit 32) between qknorm and rot_phi
	// proves that the EOF-tail seek correctly skips unparsed optimizer sections.
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		// SCFA: k=2, w=1 → D[l] has m*(w+1) = 4*2 = 8 floats
		const int    scfa_k = 2, scfa_w = 1;
		const size_t D_sz   = (size_t)RT_M * (scfa_w + 1);  // 8

		std::vector<float>         Dhost0(D_sz), Dhost1(D_sz);
		rt_fill(&Dhost0[0], D_sz, 50.0f);
		rt_fill(&Dhost1[0], D_sz, 51.0f);
		std::vector<const float*>  D_ptrs(RT_L);
		D_ptrs[0] = &Dhost0[0];
		D_ptrs[1] = &Dhost1[0];

		// QK-Norm gamma: L*nH = 2 floats
		const size_t qknorm_n = (size_t)RT_L * RT_NH;
		std::vector<float> qknormSrc(qknorm_n);
		rt_fill(&qknormSrc[0], qknorm_n, 30.0f);

		// rot_phi: L*m = 8 floats (must be LAST in file)
		const size_t rotphi_n = (size_t)RT_L * RT_M;
		std::vector<float> rotPhiSrc(rotphi_n);
		rt_fill(&rotPhiSrc[0], rotphi_n, 40.0f);

		// Dummy fp32-Adam payload: 100 floats (bit 32 — reader skips, rot_phi seeks from EOF)
		std::vector<float> dummyAdam(100, 3.14f);

		const uint32_t flags = (uint32_t)glades::chiron::CKPT_BIT_SCFA
		                     | (uint32_t)glades::chiron::CKPT_BIT_QKNORM_GAMMA
		                     | (uint32_t)glades::chiron::CKPT_BIT_FP32_ADAM
		                     | (uint32_t)glades::chiron::CKPT_BIT_ROT_PHI;

		std::string path = rt_tmppath();
		ASSERT("rt4 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt4 fopen", fp != 0);
		ASSERT("rt4 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr(flags)));
		ASSERT("rt4 write_weights", rt_write_weights(fp, src, false, false));
		ASSERT("rt4 write_scfa",    glades::chiron::chiron_write_scfa_section(fp, scfa_k, scfa_w, D_ptrs, D_sz));
		ASSERT("rt4 write_qknorm",  glades::chiron::chiron_write_qknorm_section(fp, &qknormSrc[0], qknorm_n));
		// Dummy optimizer payload (bit 32): 100 floats between qknorm and rot_phi tail.
		ASSERT("rt4 write_dummy_adam", std::fwrite(&dummyAdam[0], sizeof(float), 100, fp) == 100u);
		ASSERT("rt4 write_rotphi",  glades::chiron::chiron_write_f32_tail_section(fp, &rotPhiSrc[0], rotphi_n));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt4 load", glades::chiron::chiron_load_model(path, dims, w, err, errCode));

		rt_verify_weights(src, w, false, false);

		// SCFA
		ASSERT("rt4 scfa dLoaded", w.scfa.dLoaded);
		ASSERT("rt4 scfa k",       w.scfa.k == scfa_k);
		ASSERT("rt4 scfa w",       w.scfa.w == scfa_w);
		ASSERT("rt4 scfa D size",  (int)w.scfa.D.size() == RT_L);
		{
			std::vector<float> gotD(D_sz);
			ASSERT("rt4 scfa D[0] dl", w.scfa.D[0]->download(&gotD[0], D_sz));
			for (size_t i = 0; i < D_sz; ++i) ASSERT("rt4 D[0] val", gotD[i] == Dhost0[i]);
			ASSERT("rt4 scfa D[1] dl", w.scfa.D[1]->download(&gotD[0], D_sz));
			for (size_t i = 0; i < D_sz; ++i) ASSERT("rt4 D[1] val", gotD[i] == Dhost1[i]);
		}

		// QK-Norm gamma
		ASSERT("rt4 qknorm size", w.qknormGamma.size() == qknorm_n);
		for (size_t i = 0; i < qknorm_n; ++i)
			ASSERT("rt4 qknorm val", w.qknormGamma[i] == qknormSrc[i]);

		// rot_phi — read from EOF tail, skipping the dummy 100-float Adam payload
		ASSERT("rt4 rotphi size", w.rotPhi.size() == rotphi_n);
		for (size_t i = 0; i < rotphi_n; ++i)
			ASSERT("rt4 rotphi val", w.rotPhi[i] == rotPhiSrc[i]);

		unlink(path.c_str());
	}

	// ---- Case 5: Unknown bit 2048 → reject with errCode=4 ----
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		const uint32_t flags = 2048u;  // bit beyond CKPT_KNOWN_BITS_MASK

		std::string path = rt_tmppath();
		ASSERT("rt5 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt5 fopen", fp != 0);
		ASSERT("rt5 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr(flags)));
		ASSERT("rt5 write_weights", rt_write_weights(fp, src, false, false));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		bool ok = glades::chiron::chiron_load_model(path, dims, w, err, errCode);
		ASSERT("rt5 rejected",   !ok);
		ASSERT("rt5 errCode 4",  errCode == 4);
		ASSERT("rt5 err unknown", err.find("unknown") != std::string::npos);

		unlink(path.c_str());
	}

	// ---- Case 6: bit 512 (a_drift) + bit 1024 (rot_phi) ----
	// a_drift is the SECOND-to-last section (rot_phi is last).  The reader locates
	// a_drift by EOF arithmetic: offset = fileSize - 2*L*m*4.  Assert BOTH sections
	// load element-exact.
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		const size_t aDrift_n = (size_t)RT_L * RT_M;  // 8 floats
		std::vector<float> aDriftPayload(aDrift_n);
		rt_fill(&aDriftPayload[0], aDrift_n, 60.0f);  // known pattern

		const size_t rotphi_n = (size_t)RT_L * RT_M;  // 8 floats (must be LAST)
		std::vector<float> rotPhiSrc(rotphi_n);
		rt_fill(&rotPhiSrc[0], rotphi_n, 70.0f);

		const uint32_t flags = (uint32_t)glades::chiron::CKPT_BIT_A_DRIFT
		                     | (uint32_t)glades::chiron::CKPT_BIT_ROT_PHI;

		std::string path = rt_tmppath();
		ASSERT("rt6 tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt6 fopen", fp != 0);
		ASSERT("rt6 write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr(flags)));
		ASSERT("rt6 write_weights", rt_write_weights(fp, src, false, false));
		// a_drift written BEFORE rot_phi; reader finds it via the (bit1024?2:1) EOF arithmetic.
		ASSERT("rt6 write_adrift",  glades::chiron::chiron_write_f32_tail_section(fp, &aDriftPayload[0], aDrift_n));
		// rot_phi MUST be LAST.
		ASSERT("rt6 write_rotphi",  glades::chiron::chiron_write_f32_tail_section(fp, &rotPhiSrc[0], rotphi_n));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt6 load",      glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		ASSERT("rt6 hasADrift", w.hasADrift);
		// a_drift loaded element-exact (2-section-tail arithmetic).
		ASSERT("rt6 adrift size", w.aDrift.size() == aDrift_n);
		for (size_t i = 0; i < aDrift_n; ++i)
			ASSERT("rt6 adrift val", w.aDrift[i] == aDriftPayload[i]);
		ASSERT("rt6 rotphi size", w.rotPhi.size() == rotphi_n);
		for (size_t i = 0; i < rotphi_n; ++i)
			ASSERT("rt6 rotphi val", w.rotPhi[i] == rotPhiSrc[i]);

		unlink(path.c_str());
	}

	// ---- Case 6b: bit 512 (a_drift) with bit 1024 CLEAR — a_drift is the LAST section ----
	// Proves the (bit1024?2:1) EOF arithmetic the OTHER way: offset = fileSize - 1*L*m*4.
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		const size_t aDrift_n = (size_t)RT_L * RT_M;  // 8 floats
		std::vector<float> aDriftPayload(aDrift_n);
		rt_fill(&aDriftPayload[0], aDrift_n, 80.0f);  // known pattern

		const uint32_t flags = (uint32_t)glades::chiron::CKPT_BIT_A_DRIFT;  // NO rot_phi (bit 1024 clear)

		std::string path = rt_tmppath();
		ASSERT("rt6b tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt6b fopen", fp != 0);
		ASSERT("rt6b write_header",  glades::chiron::chiron_write_header(fp, rt_make_hdr(flags)));
		ASSERT("rt6b write_weights", rt_write_weights(fp, src, false, false));
		// a_drift is the LAST section (no rot_phi tail after it).
		ASSERT("rt6b write_adrift",  glades::chiron::chiron_write_f32_tail_section(fp, &aDriftPayload[0], aDrift_n));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt6b load",       glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		ASSERT("rt6b hasADrift",  w.hasADrift);
		ASSERT("rt6b no rotphi",  w.rotPhi.empty());
		ASSERT("rt6b adrift size", w.aDrift.size() == aDrift_n);
		for (size_t i = 0; i < aDrift_n; ++i)
			ASSERT("rt6b adrift val", w.aDrift[i] == aDriftPayload[i]);

		unlink(path.c_str());
	}

	// ---- Case 7a: Legacy CHRN v1 (weights only, no flags) ----
	{
		RTWeightData src;
		rt_fill_weights(src, false);

		std::string path = rt_tmppath();
		ASSERT("rt7a tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt7a fopen",        fp != 0);
		ASSERT("rt7a write_chrn_v1", rt_write_chrn_v1(fp, src));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt7a load",       glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		ASSERT("rt7a T",          dims.T == RT_T);
		ASSERT("rt7a dModel",     dims.dModel == RT_DM);
		rt_verify_weights(src, w, false, false);
		ASSERT("rt7a no gamma_p", w.gamma_p.empty());
		ASSERT("rt7a no adrift",  !w.hasADrift);

		unlink(path.c_str());
	}

	// ---- Case 7b: CHRN v3 with BF16_WEIGHTS | HAS_GAMMA_P ----
	{
		RTWeightData src;
		rt_fill_weights(src, true);  // with gamma_p

		const uint32_t chrnFlags = (uint32_t)glades::chiron::CHRN_BIT_BF16_WEIGHTS
		                         | (uint32_t)glades::chiron::CHRN_BIT_HAS_GAMMA_P;

		std::string path = rt_tmppath();
		ASSERT("rt7b tmppath", !path.empty());

		std::FILE* fp = std::fopen(path.c_str(), "wb");
		ASSERT("rt7b fopen",         fp != 0);
		ASSERT("rt7b write_chrn_v3", rt_write_chrn_v3(fp, src, chrnFlags));
		std::fclose(fp);

		glades::chiron::ChironModelDims     dims;
		glades::chiron::ChironModelWeights  w;
		std::string err; int errCode = 0;
		ASSERT("rt7b load",           glades::chiron::chiron_load_model(path, dims, w, err, errCode));
		rt_verify_weights(src, w, true, true);  // bf16 decode + gamma_p
		ASSERT("rt7b gamma_p size",   (int)w.gamma_p.size() == RT_L);

		unlink(path.c_str());
	}
}

// ---------------------------------------------------------------------------
// Test 4: chiron_resolve_serving — decision-table cases (3 split into 3a/3b).
//
// Tiny shape: T=64, m=4, L=2, nH=1, dH=4, V=16, dModel=4.
// We build ChironModelWeights in-memory (no file I/O) and call
// chiron_resolve_serving directly.
// ---------------------------------------------------------------------------

// Helper: allocate a GpuBuffer<float> of size n and fill with value v.
// Returns a heap-allocated pointer (caller owns).
static glades::gpu::GpuBuffer<float>* srv_alloc_buf(size_t n, float v)
{
    glades::gpu::GpuBuffer<float>* buf = new glades::gpu::GpuBuffer<float>();
    std::vector<float> data(n, v);
    bool ok = buf->allocate(n) && buf->upload(&data[0], n);
    ASSERT("srv_alloc_buf ok", ok);
    return buf;
}

// Helper: build minimal dims for resolve tests.
static glades::chiron::ChironModelDims srv_dims()
{
    glades::chiron::ChironModelDims d;
    d.T      = 64;
    d.m      = 4;
    d.L      = 2;
    d.nH     = 1;
    d.dH     = 4;
    d.V      = 16;
    d.dModel = 4;
    return d;
}

// Helper: populate per-layer weight vectors on w with minimal non-null GpuBuffers.
// Allocates Wq/Wk/Wv/Wo/gamma/beta for each layer; fills with dummy zeros.
static void srv_fill_layer_weights(glades::chiron::ChironModelWeights& w,
                                   const glades::chiron::ChironModelDims& d)
{
    const size_t Wqkv_sz = (size_t)d.m * d.dModel;
    const size_t Wo_sz   = (size_t)d.dModel * d.m;
    for (int l = 0; l < d.L; ++l)
    {
        w.Wq.push_back(srv_alloc_buf(Wqkv_sz, 0.0f));
        w.Wk.push_back(srv_alloc_buf(Wqkv_sz, 0.0f));
        w.Wv.push_back(srv_alloc_buf(Wqkv_sz, 0.0f));
        w.Wo.push_back(srv_alloc_buf(Wo_sz,   0.0f));
        w.gamma.push_back(srv_alloc_buf((size_t)d.m, 1.0f));
        w.beta.push_back( srv_alloc_buf((size_t)d.m, 0.0f));
    }
}

void CHIRONResolveServingTest()
{
    // ---- Case 1: rotPhi present + !o.whiscCoupling → return 7 ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // Put something in rotPhi to signal WhiSC checkpoint.
        w.rotPhi.assign((size_t)dims.L * dims.m, 0.01f);

        glades::chiron::ChironServingOverrides o;
        o.whiscCoupling = false;  // NOT passed

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case1 returns 7", rc == 7);
        ASSERT("case1 err not empty", !err.empty());
    }

    // ---- Case 2: rotPhi empty + o.whiscCoupling → return 7 ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // rotPhi is empty (no WhiSC state).

        glades::chiron::ChironServingOverrides o;
        o.whiscCoupling = true;  // --whisc-coupling passed but checkpoint lacks it

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case2 returns 7", rc == 7);
        ASSERT("case2 err not empty", !err.empty());
    }

    // ---- Case 3a: hasADrift + NONZERO a_drift → return 7, err mentions a_drift ----
    // A genuinely drift-TRAINED checkpoint must be refused: the eval forward does
    // not apply per-layer drift, so serving would be silently wrong.
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        w.hasADrift = true;
        // L*m a_drift with at least one nonzero element (exact != 0.0f scan).
        w.aDrift.assign((size_t)dims.L * dims.m, 0.0f);
        w.aDrift[3] = 0.7f;  // a trained value is far from zero

        glades::chiron::ChironServingOverrides o;

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case3a returns 7", rc == 7);
        ASSERT("case3a err mentions a_drift", err.find("a_drift") != std::string::npos);
    }

    // ---- Case 3b: hasADrift + ALL-ZERO a_drift → return 0, err untouched ----
    // The zero-init a_drift that --whisc-coupling co-allocates is an exact no-op;
    // serving proceeds (this is the production PIED / WhiSC-D case).
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        w.hasADrift = true;
        w.aDrift.assign((size_t)dims.L * dims.m, 0.0f);  // all-zero, sized L*m

        glades::chiron::ChironServingOverrides o;
        o.scfaForceMode = -1;  // force SCFA off so we don't fail on B alloc

        glades::chiron::ChironServingConfig cfg;
        std::string err = "SENTINEL";  // prove the all-zero path does NOT touch err
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case3b returns 0", rc == 0);
        ASSERT("case3b err untouched", err == "SENTINEL");
    }

    // ---- Case 4: gamma_p empty + fuse unset → cfg.fuseAttnPerLayer == false ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // gamma_p empty, scfa.dLoaded=false (no SCFA)

        glades::chiron::ChironServingOverrides o;
        o.fuseAttnPerLayer = -1;  // unset
        o.scfaForceMode    = -1;  // force SCFA off so we don't fail on B alloc

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case4 returns 0", rc == 0);
        ASSERT("case4 fuseAttnPerLayer disabled", !cfg.fuseAttnPerLayer);
    }

    // ---- Case 5: gamma_p empty + scfa.dLoaded + !o.fuseAttnReln → cfg.fuseAttnReln == true ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // gamma_p empty, scfa.dLoaded=true
        w.scfa.dLoaded = true;
        w.scfa.k = 4;
        w.scfa.w = 1;
        // Allocate D for the loaded scfa
        const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
        for (int l = 0; l < dims.L; ++l)
            w.scfa.D.push_back(srv_alloc_buf(D_sz, 0.0f));

        glades::chiron::ChironServingOverrides o;
        o.fuseAttnReln  = false;
        o.scfaForceMode = 1;  // --scfa (force on with loaded D)

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case5 returns 0", rc == 0);
        ASSERT("case5 fuseAttnReln auto-enabled", cfg.fuseAttnReln);
    }

    // ---- Case 6: WhiSC + gamma_p at dead init + scfa.dLoaded + fuse unset →
    //              fuseAttnPerLayer=false, fuseAttnReln=true ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // rotPhi non-empty (WhiSC checkpoint)
        w.rotPhi.assign((size_t)dims.L * dims.m, 0.01f);
        // gamma_p at dead init: gamma_p[i]=1.0, beta_p[i]=0.0
        for (int l = 0; l < dims.L; ++l)
        {
            w.gamma_p.push_back(srv_alloc_buf((size_t)dims.m, 1.0f));
            w.beta_p.push_back( srv_alloc_buf((size_t)dims.m, 0.0f));
        }
        // SCFA loaded
        w.scfa.dLoaded = true;
        w.scfa.k = 4;
        w.scfa.w = 1;
        const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
        for (int l = 0; l < dims.L; ++l)
            w.scfa.D.push_back(srv_alloc_buf(D_sz, 0.0f));

        glades::chiron::ChironServingOverrides o;
        o.whiscCoupling    = true;   // matches checkpoint
        o.fuseAttnPerLayer = -1;     // unset
        o.scfaForceMode    = 1;      // --scfa

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case6 returns 0", rc == 0);
        // Dead-gamma_p guard: per-layer fuse disabled, reln-fuse enabled
        ASSERT("case6 fuseAttnPerLayer false", !cfg.fuseAttnPerLayer);
        ASSERT("case6 fuseAttnReln true", cfg.fuseAttnReln);
    }

    // ---- Case 7: Same but gamma_p[0][0]=1.5 (trained) → per-layer fuse kept ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // rotPhi non-empty (WhiSC checkpoint)
        w.rotPhi.assign((size_t)dims.L * dims.m, 0.01f);
        // gamma_p layer 0 has value 1.5 (trained, NOT dead init)
        std::vector<float> gp0((size_t)dims.m, 1.5f);
        glades::gpu::GpuBuffer<float>* gp0_buf = new glades::gpu::GpuBuffer<float>();
        bool ok0 = gp0_buf->allocate((size_t)dims.m) && gp0_buf->upload(&gp0[0], (size_t)dims.m);
        ASSERT("case7 gp0 upload", ok0);
        w.gamma_p.push_back(gp0_buf);
        for (int l = 1; l < dims.L; ++l)
            w.gamma_p.push_back(srv_alloc_buf((size_t)dims.m, 1.0f));
        for (int l = 0; l < dims.L; ++l)
            w.beta_p.push_back(srv_alloc_buf((size_t)dims.m, 0.0f));
        // SCFA loaded
        w.scfa.dLoaded = true;
        w.scfa.k = 4;
        w.scfa.w = 1;
        const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
        for (int l = 0; l < dims.L; ++l)
            w.scfa.D.push_back(srv_alloc_buf(D_sz, 0.0f));

        glades::chiron::ChironServingOverrides o;
        o.whiscCoupling    = true;
        o.fuseAttnPerLayer = -1;  // unset
        o.scfaForceMode    = 1;   // --scfa

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case7 returns 0", rc == 0);
        // Trained gamma_p: per-layer fuse kept ON (guard path: warn, not switch)
        ASSERT("case7 fuseAttnPerLayer true", cfg.fuseAttnPerLayer);
    }

    // ---- Case 8: qknormGamma sized L*nH + !o.qkNorm → auto-enable + exact-γ scale ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // Exact per-head gamma: L*nH = 2*1 = 2 floats
        w.qknormGamma.resize((size_t)dims.L * dims.nH);
        w.qknormGamma[0] = 2.0f;
        w.qknormGamma[1] = 3.0f;

        glades::chiron::ChironServingOverrides o;
        o.qkNorm      = false;   // NOT explicitly enabled
        o.scfaForceMode = -1;   // no SCFA

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case8 returns 0", rc == 0);
        ASSERT("case8 qkNorm auto-enabled", cfg.qkNorm);
        ASSERT("case8 gammaScale size", (int)cfg.qknormGammaScale.size() == dims.L * dims.nH);
        const float sqrtDh = std::sqrt((float)dims.dH);  // sqrt(4)=2
        ASSERT("case8 gammaScale[0]", cfg.qknormGammaScale[0] == w.qknormGamma[0] * sqrtDh);
        ASSERT("case8 gammaScale[1]", cfg.qknormGammaScale[1] == w.qknormGamma[1] * sqrtDh);
    }

    // ---- Case 9: qknormGamma empty + o.qkNorm → approx gamma=log2(T) used ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // qknormGamma empty (no bit 256 in checkpoint)

        glades::chiron::ChironServingOverrides o;
        o.qkNorm      = true;   // operator explicitly passes --qk-norm
        o.qkNormGamma = 0.0f;  // auto → will use log2(T)=6
        o.scfaForceMode = -1;  // no SCFA

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case9 returns 0", rc == 0);
        ASSERT("case9 qkNorm enabled", cfg.qkNorm);
        ASSERT("case9 gammaScale size", (int)cfg.qknormGammaScale.size() == dims.L * dims.nH);
        const float expectedGamma = std::log((float)dims.T) / std::log(2.0f);  // log2(64)=6
        const float sqrtDh = std::sqrt((float)dims.dH);
        const float expected = expectedGamma * sqrtDh;
        // Allow small floating point tolerance
        ASSERT("case9 gammaScale approx", std::fabs(cfg.qknormGammaScale[0] - expected) < 1e-5f);
    }

    // ---- Case 10: scfa.dLoaded + auto mode → useScfa=true;
    //              scfaForceMode==-1 → false ----
    {
        // Sub-case 10a: auto mode with dLoaded → useScfa=true
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        w.scfa.dLoaded = true;
        w.scfa.k = 4;
        w.scfa.w = 1;
        const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
        for (int l = 0; l < dims.L; ++l)
            w.scfa.D.push_back(srv_alloc_buf(D_sz, 0.0f));

        glades::chiron::ChironServingOverrides o;
        o.scfaForceMode = 0;  // auto

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case10a returns 0", rc == 0);
        ASSERT("case10a useScfa true", cfg.useScfa);
    }
    {
        // Sub-case 10b: scfaForceMode==-1 → useScfa=false
        glades::chiron::ChironModelDims   dims = srv_dims();
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        w.scfa.dLoaded = true;
        w.scfa.k = 4;
        w.scfa.w = 1;
        const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
        for (int l = 0; l < dims.L; ++l)
            w.scfa.D.push_back(srv_alloc_buf(D_sz, 0.0f));

        glades::chiron::ChironServingOverrides o;
        o.scfaForceMode = -1;  // --no-scfa

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case10b returns 0", rc == 0);
        ASSERT("case10b useScfa false", !cfg.useScfa);
    }

    // ---- Case 11: o.seqLen=4 (< T=64) → dims.T == 4 ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();  // T=64
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);

        glades::chiron::ChironServingOverrides o;
        o.seqLen        = 4;
        o.scfaForceMode = -1;  // no SCFA

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case11 returns 0", rc == 0);
        ASSERT("case11 dims.T == 4", dims.T == 4);
    }

    // ---- Case 12: SCFA on with no loaded D → D allocated+zeroed, B allocated,
    //              k/w defaulted to T/16 (min 4) and 8; returns 0 ----
    {
        glades::chiron::ChironModelDims   dims = srv_dims();  // T=64
        glades::chiron::ChironModelWeights w;
        srv_fill_layer_weights(w, dims);
        // scfa.dLoaded = false, D is empty

        glades::chiron::ChironServingOverrides o;
        o.scfaForceMode    = 1;   // --scfa: force on
        o.scfaKOverride    = 0;   // use defaults
        o.scfaWOverride    = -1;  // use defaults

        glades::chiron::ChironServingConfig cfg;
        std::string err;
        int rc = glades::chiron::chiron_resolve_serving(dims, w, o, cfg, err);
        ASSERT("case12 returns 0", rc == 0);
        ASSERT("case12 useScfa true", cfg.useScfa);
        // k defaults to T/16 = 64/16 = 4
        ASSERT("case12 scfa.k == 4", w.scfa.k == 4);
        // w defaults to 8
        ASSERT("case12 scfa.w == 8", w.scfa.w == 8);
        // B should be allocated
        ASSERT("case12 B allocated", w.scfa.B.allocated());
        // D should be allocated for each layer
        ASSERT("case12 D size", (int)w.scfa.D.size() == dims.L);
        for (int l = 0; l < dims.L; ++l)
        {
            ASSERT("case12 D[l] not null", w.scfa.D[l] != 0);
            ASSERT("case12 D[l] allocated", w.scfa.D[l]->allocated());
            // Verify it was zeroed
            const size_t D_sz = (size_t)dims.m * (w.scfa.w + 1);
            std::vector<float> dbuf(D_sz, 99.0f);
            ASSERT("case12 D[l] download", w.scfa.D[l]->download(&dbuf[0], D_sz));
            for (size_t i = 0; i < D_sz; ++i)
                ASSERT("case12 D[l] zero", dbuf[i] == 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: chiron_eval_forward orchestration parity.
//
// The reference below (EvalRefScratch / ref_scfa_shear / ref_forward) is an
// INDEPENDENT transcription of chiron_infer.cpp's Scratch / scfa_shear_infer /
// forwardInfer (lines 359-740, glades-trainer) — it is NOT a call into the
// library function under test.  It composes the same glades::gpu primitives in
// the same order with the same arguments and buffers; the test then asserts
// chiron_eval_forward produces BIT-IDENTICAL logits.  This targets the actual
// risk class of this task: orchestration bugs (wrong order / arg / buffer).
// Kernel *math* is already covered by the other chiron suites.
//
// Shape: T=8, m=4, L=2, nH=1, dH=4, V=16, dModel=4.
// ---------------------------------------------------------------------------

static const int EV_T  = 8;
static const int EV_M  = 4;
static const int EV_L  = 2;
static const int EV_NH = 1;
static const int EV_DH = 4;
static const int EV_V  = 16;
static const int EV_DM = 4;  // nH * dH

// Independent copy of chiron_infer.cpp:359-457 Scratch (adapted to the ml types).
struct EvalRefScratch
{
	glades::gpu::GpuBuffer<int>   d_tokens;
	glades::gpu::GpuBuffer<float> q, p, q_tmp, stats;
	glades::gpu::GpuBuffer<float> sQ, sK, sV, sO, scratch_P, logits;
	glades::gpu::GpuBuffer<float> p_norm, stats_p;
	glades::gpu::GpuBuffer<float> scfa_qcompr, scfa_qpar, scfa_qperp, scfa_yperp, scfa_ycompr, scfa_ypar;
	glades::gpu::GpuBuffer<float> scfa_inner_p, scfa_inner_sQ, scfa_inner_sK, scfa_inner_sV, scfa_inner_sO, scfa_inner_sP;
	glades::gpu::GpuBuffer<float> qknorm_invNorm, qknorm_gamma_scale;
	glades::gpu::GpuBuffer<float> rot_a, rot_c, whisc_Pbar, whisc_Qbar, whisc_a;

	bool allocate(const glades::chiron::ChironModelDims& d,
	              const glades::chiron::ChironServingConfig& cfg, int scfaK)
	{
		const int T = d.T, m = d.m, L = d.L, dModel = d.dModel;
		if (!d_tokens.allocate(T)) return false;
		if (!q.allocate((size_t)T * m)) return false;
		if (!p.allocate((size_t)T * m)) return false;
		if (!q_tmp.allocate((size_t)T * m)) return false;
		if (!stats.allocate((size_t)L * T * 2u)) return false;
		if (!logits.allocate((size_t)T * d.V)) return false;
		if (!p_norm.allocate((size_t)T * m)) return false;
		if (!stats_p.allocate((size_t)L * T * 2u)) return false;
		if (cfg.useScfa)
		{
			const size_t Tm  = (size_t)T * (size_t)m;
			const size_t km  = (size_t)scfaK * (size_t)m;
			const size_t kdM = (size_t)scfaK * (size_t)dModel;
			const size_t Skk = (size_t)d.nH * (size_t)scfaK * (size_t)scfaK;
			if (!scfa_qcompr.allocate(km)) return false;
			if (!scfa_qpar.allocate(Tm)) return false;
			if (!scfa_qperp.allocate(Tm)) return false;
			if (!scfa_yperp.allocate(Tm)) return false;
			if (!scfa_ycompr.allocate(km)) return false;
			if (!scfa_ypar.allocate(Tm)) return false;
			if (!scfa_inner_p.allocate(km)) return false;
			if (!scfa_inner_sQ.allocate(kdM)) return false;
			if (!scfa_inner_sK.allocate(kdM)) return false;
			if (!scfa_inner_sV.allocate(kdM)) return false;
			if (!scfa_inner_sO.allocate(kdM)) return false;
			if (!scfa_inner_sP.allocate(Skk)) return false;
			if (!qknorm_invNorm.allocate((size_t)scfaK * d.nH)) return false;
			if (!qknorm_gamma_scale.allocate((size_t)d.L * d.nH)) return false;
		}
		else
		{
			if (!sQ.allocate((size_t)T * dModel)) return false;
			if (!sK.allocate((size_t)T * dModel)) return false;
			if (!sV.allocate((size_t)T * dModel)) return false;
			if (!sO.allocate((size_t)T * dModel)) return false;
			if (!scratch_P.allocate((size_t)d.nH * T * T)) return false;
		}
		if (cfg.whiscCoupling)
		{
			if (!rot_a.allocate((size_t)m)) return false;
			if (!rot_c.allocate((size_t)m)) return false;
			if (!whisc_Pbar.allocate((size_t)m)) return false;
			if (!whisc_Qbar.allocate((size_t)m)) return false;
			if (!whisc_a.allocate((size_t)m)) return false;
			std::vector<float> ones((size_t)m, 1.0f);
			if (!whisc_Pbar.upload(&ones[0], (size_t)m)) return false;
			if (!whisc_Qbar.upload(&ones[0], (size_t)m)) return false;
		}
		if (!cfg.qknormGammaScale.empty())
		{
			if (!qknorm_gamma_scale.upload(&cfg.qknormGammaScale[0], cfg.qknormGammaScale.size()))
				return false;
		}
		return true;
	}
};

// Independent transcription of chiron_infer.cpp:484-577 (scfa_shear_infer),
// QK-Norm branch (526-548).  Same primitives, same order, same args.
static bool ref_scfa_shear(EvalRefScratch& s, const glades::chiron::ChironScfaState& scfa,
                           const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                           int layer, int T, int m, int nH, int dH, bool invert, bool qkNorm)
{
	const int k = scfa.k;
	const int w = scfa.w;
	const int dModel = nH * dH;
	const size_t Tm = (size_t)T * (size_t)m;
	const size_t km = (size_t)k * (size_t)m;

	if (!glades::gpu::sgemm_rowmajor_atb(k, m, T, 1.0f,
	        scfa.B.data(), k, s.q.data(), m, 0.0f, s.scfa_qcompr.data(), m)) return false;
	if (!glades::gpu::sgemm_rowmajor(T, m, k, 1.0f,
	        scfa.B.data(), k, s.scfa_qcompr.data(), m, 0.0f, s.scfa_qpar.data(), m)) return false;
	glades::gpu::device_memcpy_d2d(s.scfa_qperp.data(), s.q.data(), sizeof(float) * Tm);
	if (!glades::gpu::axpy(-1.0f, s.scfa_qpar.data(), s.scfa_qperp.data(), (int)Tm)) return false;
	if (!glades::gpu::scfa_depthwise_causal_conv_fwd(
	        s.scfa_qperp.data(), scfa.D[layer]->data(), T, m, w, s.scfa_yperp.data())) return false;
	if (!s.scfa_inner_p.zero()) return false;
	if (qkNorm)
	{
		if (!glades::gpu::sgemm_rowmajor(k, dModel, m, 1.0f,
		        s.scfa_qcompr.data(), m, Wq, dModel, 0.0f, s.scfa_inner_sQ.data(), dModel)) return false;
		if (!glades::gpu::sgemm_rowmajor(k, dModel, m, 1.0f,
		        s.scfa_qcompr.data(), m, Wk, dModel, 0.0f, s.scfa_inner_sK.data(), dModel)) return false;
		if (!glades::gpu::sgemm_rowmajor(k, dModel, m, 1.0f,
		        s.scfa_qcompr.data(), m, Wv, dModel, 0.0f, s.scfa_inner_sV.data(), dModel)) return false;
		if (!glades::gpu::qknorm_forward_gpu(s.scfa_inner_sQ.data(), s.qknorm_invNorm.data(), k, nH, dH, 1e-6f)) return false;
		if (!glades::gpu::qknorm_forward_gpu(s.scfa_inner_sK.data(), s.qknorm_invNorm.data(), k, nH, dH, 1e-6f)) return false;
		if (!glades::gpu::scale_q_per_head(s.scfa_inner_sQ.data(), s.qknorm_gamma_scale.data() + (size_t)layer * nH, k, nH, dH)) return false;
		if (!glades::gpu::flash_attention_cublas_tiled(
		        s.scfa_inner_sQ.data(), s.scfa_inner_sK.data(), s.scfa_inner_sV.data(),
		        k, nH, dH, dModel, true, s.scfa_inner_sO.data(), s.scfa_inner_sP.data())) return false;
		if (!glades::gpu::sgemm_rowmajor(k, m, dModel, 1.0f,
		        s.scfa_inner_sO.data(), dModel, Wo, m, 0.0f, s.scfa_inner_p.data(), m)) return false;
	}
	else if (!glades::gpu::chiron_attention_shear_tiled(
	        s.scfa_qcompr.data(), s.scfa_inner_p.data(), Wq, Wk, Wv, Wo,
	        k, m, nH, dH, true, false,
	        s.scfa_inner_sQ.data(), s.scfa_inner_sK.data(),
	        s.scfa_inner_sV.data(), s.scfa_inner_sO.data(), s.scfa_inner_sP.data())) return false;
	glades::gpu::device_memcpy_d2d(s.scfa_ycompr.data(), s.scfa_inner_p.data(), sizeof(float) * km);
	if (!glades::gpu::sgemm_rowmajor(T, m, k, 1.0f,
	        scfa.B.data(), k, s.scfa_ycompr.data(), m, 0.0f, s.scfa_ypar.data(), m)) return false;
	const float sign = invert ? -1.0f : 1.0f;
	if (!glades::gpu::axpy(1.0f, s.scfa_yperp.data(), s.scfa_ypar.data(), (int)Tm)) return false;
	if (!glades::gpu::axpy(sign, s.scfa_ypar.data(), s.p.data(), (int)Tm)) return false;
	return true;
}

// Independent transcription of chiron_infer.cpp:580-740 (forwardInfer), minus
// the env-gated CHIRON_DBG blocks.  rotPhiRef supplies the per-layer angle
// buffers (independent of the scratch under test).
static bool ref_forward(const glades::chiron::ChironModelDims& d, EvalRefScratch& s,
                        const glades::chiron::ChironModelWeights& w,
                        const glades::chiron::ChironServingConfig& cfg,
                        const std::vector<glades::gpu::GpuBuffer<float>*>& rotPhiRef)
{
	const int T = d.T, m = d.m, V = d.V, L = d.L, nH = d.nH, dH = d.dH;

	const bool applyFuse = cfg.fuseAttnPerLayer
	                       && !w.gamma_p.empty() && !w.beta_p.empty()
	                       && (int)w.gamma_p.size() == L && (int)w.beta_p.size() == L;
	// Reference keeps the plain 1/sqrt(L) (chiron_infer's value); applyFuse is
	// false in every parity case so fuseAlpha is never used.
	const float fuseAlpha = applyFuse ? (1.0f / std::sqrt((float)L)) : 0.0f;

	if (!glades::gpu::embedding_gather(w.E.data(), s.d_tokens.data(), T, V, m, s.q.data())) return false;
	if (!s.p.zero()) return false;

	for (int l = 0; l < L; ++l)
	{
		if (w.scfa.present)
		{
			if (!ref_scfa_shear(s, w.scfa,
			                    w.Wq[l]->data(), w.Wk[l]->data(), w.Wv[l]->data(), w.Wo[l]->data(),
			                    l, T, m, nH, dH, false, cfg.qkNorm)) return false;
		}
		else if (!glades::gpu::chiron_attention_shear_tiled(
		        s.q.data(), s.p.data(),
		        w.Wq[l]->data(), w.Wk[l]->data(), w.Wv[l]->data(), w.Wo[l]->data(),
		        T, m, nH, dH, true, false,
		        s.sQ.data(), s.sK.data(), s.sV.data(), s.sO.data(), s.scratch_P.data())) return false;

		if (cfg.fuseAttnReln && l == L - 1)
		{
			if (!glades::gpu::axpy(1.0f, s.p.data(), s.q.data(), T * m)) return false;
		}

		if (applyFuse)
		{
			if (!glades::gpu::chiron_reln_forward(
			        s.p.data(), s.p_norm.data(), s.stats_p.data() + (size_t)l * T * 2u,
			        w.gamma_p[l]->data(), w.beta_p[l]->data(), T, m, cfg.epsReln)) return false;
			if (!glades::gpu::axpy(fuseAlpha, s.p_norm.data(), s.q.data(), T * m)) return false;
		}

		if (cfg.whiscCoupling)
		{
			if (!glades::gpu::chiron_rot_coeffs(
			        rotPhiRef[l]->data(), cfg.rotThetaMax, 1.0f, m,
			        s.rot_a.data(), s.rot_c.data())) return false;
			if (!glades::gpu::chiron_whisc_update_stats(
			        s.q.data(), s.p.data(), T, m, 1.0f, 1e-12f, cfg.whiscClamp,
			        s.whisc_Pbar.data(), s.whisc_Qbar.data(), s.whisc_a.data())) return false;
			if (!glades::gpu::chiron_whisc_fold_coeffs(
			        s.rot_a.data(), s.rot_c.data(), s.whisc_a.data(), m)) return false;
			if (!glades::gpu::chiron_rot_forward(
			        s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(), 1.0f, T, m)) return false;
		}

		if (!glades::gpu::chiron_reln_forward(
		        s.q.data(), s.q_tmp.data(), s.stats.data() + (size_t)l * T * 2u,
		        w.gamma[l]->data(), w.beta[l]->data(), T, m, cfg.epsReln)) return false;
		glades::gpu::device_memcpy_d2d(s.q.data(), s.q_tmp.data(), sizeof(float) * T * m);
	}

	if (!glades::gpu::sgemm_rowmajor_abt(T, V, m, 1.0f,
	        s.q.data(), m, w.E.data(), m, 0.0f, s.logits.data(), V)) return false;
	return true;
}

// Deterministic weight/embedding fill for the parity cases.
static glades::gpu::GpuBuffer<float>* ev_alloc(size_t n, float base, float step)
{
	glades::gpu::GpuBuffer<float>* buf = new glades::gpu::GpuBuffer<float>();
	std::vector<float> h(n);
	for (size_t i = 0; i < n; ++i) h[i] = std::sin(step * (float)i + base);
	bool ok = buf->allocate(n) && buf->upload(&h[0], n);
	ASSERT("ev_alloc", ok);
	return buf;
}

// Fill E + per-layer Wq/Wk/Wv/Wo/gamma/beta on w.  No gamma_p (fuse-per-layer off).
static void ev_fill_core_weights(glades::chiron::ChironModelWeights& w,
                                 const glades::chiron::ChironModelDims& d)
{
	const size_t Esize   = (size_t)d.V * d.m;
	const size_t Wqkv_sz = (size_t)d.m * d.dModel;
	const size_t Wo_sz   = (size_t)d.dModel * d.m;
	std::vector<float> e(Esize);
	for (size_t i = 0; i < Esize; ++i) e[i] = std::sin(0.05f * (float)i);
	ASSERT("ev E alloc", w.E.allocate(Esize) && w.E.upload(&e[0], Esize));
	for (int l = 0; l < d.L; ++l)
	{
		float off = 0.3f + (float)l * 0.7f;
		w.Wq.push_back(ev_alloc(Wqkv_sz, off + 0.0f, 0.017f));
		w.Wk.push_back(ev_alloc(Wqkv_sz, off + 0.1f, 0.019f));
		w.Wv.push_back(ev_alloc(Wqkv_sz, off + 0.2f, 0.023f));
		w.Wo.push_back(ev_alloc(Wo_sz,   off + 0.3f, 0.029f));
		w.gamma.push_back(ev_alloc((size_t)d.m, off + 0.4f, 0.03f));
		w.beta.push_back( ev_alloc((size_t)d.m, off + 0.5f, 0.04f));
	}
}

static glades::chiron::ChironModelDims ev_dims()
{
	glades::chiron::ChironModelDims d;
	d.T = EV_T; d.m = EV_M; d.L = EV_L; d.nH = EV_NH; d.dH = EV_DH; d.V = EV_V; d.dModel = EV_DM;
	return d;
}

// Run ref_forward + chiron_eval_forward and assert the logits are bit-identical.
static void ev_compare(const char* tag,
                       const glades::chiron::ChironModelDims& d,
                       glades::chiron::ChironModelWeights& w,
                       const glades::chiron::ChironServingConfig& cfg,
                       const std::vector<int>& tokens,
                       int scfaK)
{
	const size_t Ln = (size_t)d.T * d.V;

	// --- Reference (independent transcription) ---
	std::vector<glades::gpu::GpuBuffer<float>*> rotPhiRef;
	if (cfg.whiscCoupling)
	{
		rotPhiRef.assign(d.L, (glades::gpu::GpuBuffer<float>*)0);
		for (int l = 0; l < d.L; ++l)
		{
			rotPhiRef[l] = new glades::gpu::GpuBuffer<float>();
			ASSERT("ev rotPhiRef alloc", rotPhiRef[l]->allocate((size_t)d.m));
			ASSERT("ev rotPhiRef upload", rotPhiRef[l]->upload(&w.rotPhi[(size_t)l * d.m], (size_t)d.m));
		}
	}
	EvalRefScratch rs;
	ASSERT("ev ref scratch alloc", rs.allocate(d, cfg, scfaK));
	ASSERT("ev ref tokens", rs.d_tokens.upload(&tokens[0], (size_t)d.T));
	ASSERT("ev ref forward", ref_forward(d, rs, w, cfg, rotPhiRef));
	std::vector<float> refLogits(Ln);
	ASSERT("ev ref download", rs.logits.download(&refLogits[0], Ln));
	for (size_t l = 0; l < rotPhiRef.size(); ++l) delete rotPhiRef[l];

	// --- Actual (library eval forward) ---
	glades::chiron::ChironEvalScratch es;
	ASSERT("ev scratch alloc", es.allocate(d, w, cfg));
	ASSERT("ev tokens", es.d_tokens.upload(&tokens[0], (size_t)d.T));
	ASSERT("ev forward", glades::chiron::chiron_eval_forward(d, w, cfg, es));
	std::vector<float> gotLogits(Ln);
	ASSERT("ev got download", es.logits.download(&gotLogits[0], Ln));

	// --- Bit-identical assertion ---
	for (size_t i = 0; i < Ln; ++i)
		ASSERT(tag, gotLogits[i] == refLogits[i]);
}

void CHIRONEvalForwardParityTest()
{
	std::vector<int> tokens((size_t)EV_T);
	for (int i = 0; i < EV_T; ++i) tokens[i] = (i * 2 + 1) % EV_V;

	// ---- Case 1: dense path, fuseAttnReln=true ----
	{
		glades::chiron::ChironModelDims d = ev_dims();
		glades::chiron::ChironModelWeights w;
		ev_fill_core_weights(w, d);

		glades::chiron::ChironServingConfig cfg;
		cfg.fuseAttnReln = true;
		cfg.epsReln = 1e-4f;
		// useScfa=false, qkNorm=false, whiscCoupling=false, fuseAttnPerLayer=false

		ev_compare("case1 dense fuse-reln bit-identical", d, w, cfg, tokens, 0);
	}

	// ---- Case 2: dense path + whiscCoupling=true (tiny nonzero rotPhi) ----
	{
		glades::chiron::ChironModelDims d = ev_dims();
		glades::chiron::ChironModelWeights w;
		ev_fill_core_weights(w, d);
		// Nonzero rot_phi angles (L*m), small so the coupling is a gentle rotation.
		w.rotPhi.resize((size_t)d.L * d.m);
		for (size_t i = 0; i < w.rotPhi.size(); ++i) w.rotPhi[i] = 0.05f + 0.01f * (float)i;

		glades::chiron::ChironServingConfig cfg;
		cfg.fuseAttnReln  = true;
		cfg.whiscCoupling = true;
		cfg.rotThetaMax   = 0.07f;
		cfg.whiscClamp    = 8.0f;
		cfg.epsReln       = 1e-4f;

		ev_compare("case2 dense+whisc bit-identical", d, w, cfg, tokens, 0);
	}

	// ---- Case 3: SCFA path + qkNorm (T=8, k=4, w=1) ----
	{
		glades::chiron::ChironModelDims d = ev_dims();
		glades::chiron::ChironModelWeights w;
		ev_fill_core_weights(w, d);

		// SCFA state: k=4, w=1; real DCT basis; small nonzero D.
		const int k = 4, sw = 1;
		w.scfa.present = true;
		w.scfa.dLoaded = true;
		w.scfa.k = k;
		w.scfa.w = sw;
		const size_t B_sz = (size_t)d.T * k;
		ASSERT("case3 B alloc", w.scfa.B.allocate(B_sz));
		ASSERT("case3 B init",  glades::gpu::scfa_dct_basis_init(w.scfa.B.data(), d.T, k));
		const size_t D_sz = (size_t)d.m * (sw + 1);
		for (int l = 0; l < d.L; ++l)
		{
			glades::gpu::GpuBuffer<float>* Dl = new glades::gpu::GpuBuffer<float>();
			std::vector<float> dh(D_sz);
			for (size_t i = 0; i < D_sz; ++i) dh[i] = 0.02f * std::sin(0.3f * (float)i + 0.11f * (float)l);
			ASSERT("case3 D alloc", Dl->allocate(D_sz) && Dl->upload(&dh[0], D_sz));
			w.scfa.D.push_back(Dl);
		}

		glades::chiron::ChironServingConfig cfg;
		cfg.useScfa      = true;
		cfg.qkNorm       = true;
		cfg.fuseAttnReln = true;   // SCFA production serving path
		cfg.epsReln      = 1e-4f;
		// Exact per-head gamma scale (L*nH = 2), nonzero.
		cfg.qknormGammaScale.resize((size_t)d.L * d.nH);
		for (size_t i = 0; i < cfg.qknormGammaScale.size(); ++i)
			cfg.qknormGammaScale[i] = 1.5f + 0.25f * (float)i;

		ev_compare("case3 scfa+qknorm bit-identical", d, w, cfg, tokens, k);
	}
}

// ---------------------------------------------------------------------------
// Aggregate entry.
// ---------------------------------------------------------------------------

void CHIRONModelUnitTest()
{
	CHIRONCkptBlockCodecTest();
	CHIRONCkptBf16RneDiscriminatingTest();
	CHIRONCkptRoundtripTest();
	CHIRONResolveServingTest();
	CHIRONEvalForwardParityTest();
}
