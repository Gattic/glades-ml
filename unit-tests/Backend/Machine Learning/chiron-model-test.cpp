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

#include <cstdio>
#include <vector>
#include <stdint.h>

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
// Aggregate entry.
// ---------------------------------------------------------------------------

void CHIRONModelUnitTest()
{
	CHIRONCkptBlockCodecTest();
	CHIRONCkptBf16RneDiscriminatingTest();
}
