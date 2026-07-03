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
// Aggregate entry.
// ---------------------------------------------------------------------------

void CHIRONModelUnitTest()
{
	CHIRONCkptBlockCodecTest();
}
