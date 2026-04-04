// Reusable tiled GEMM helpers shared across translation units.
//
// When CBLAS is available (define GLADES_HAVE_CBLAS and link -lcblas),
// each function delegates to cblas_sgemm for optimal performance.
// Otherwise falls back to cache-blocked loops with SIMD-friendly inner
// kernels provided by transformer_kernels.h, parallelized across rows
// using the engine's ThreadPool.
//
// All functions are inline to prevent ODR violations when this header
// is included from multiple translation units.
// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
#pragma once

#include "transformer_kernels.h"
#include "glades_thread_pool.h"
#include <cstring>

#ifdef GLADES_HAVE_CBLAS
#include <cblas.h>
#endif

namespace glades {
namespace gemm {

using glades::transformer_kernels::dot_f32;
using glades::transformer_kernels::axpy_f32;

// Minimum FLOPs before spawning threads (avoid overhead for tiny GEMMs).
static const unsigned long long kParallelThreshold = 200000ULL;

// ---- Parallel dispatch contexts ----

struct AtbCtx
{
	float* C;
	const float* A;
	const float* B;
	unsigned int M, K, N;
	float alpha;
};

static inline void atb_body(void* ud, unsigned int begin, unsigned int end)
{
	const AtbCtx& c = *static_cast<const AtbCtx*>(ud);
	const unsigned int TILE_K = 64u;
	const unsigned int TILE_N = 64u;

	for (unsigned int m = begin; m < end; ++m)
	{
		float* crow = c.C + static_cast<size_t>(m) * c.N;
		std::memset(crow, 0, static_cast<size_t>(c.N) * sizeof(float));
		for (unsigned int kb = 0; kb < c.K; kb += TILE_K)
		{
			const unsigned int ke = (kb + TILE_K < c.K) ? (kb + TILE_K) : c.K;
			for (unsigned int nb = 0; nb < c.N; nb += TILE_N)
			{
				const unsigned int ne = (nb + TILE_N < c.N) ? (nb + TILE_N) : c.N;
				const unsigned int nlen = ne - nb;
				for (unsigned int k = kb; k < ke; ++k)
				{
					const float a = c.alpha * c.A[static_cast<size_t>(k) * c.M + m];
					axpy_f32(crow + nb, &c.B[static_cast<size_t>(k) * c.N + nb], a, nlen);
				}
			}
		}
	}
}

struct AbAccumCtx
{
	float* C;
	const float* A;
	const float* B;
	unsigned int M, R, N;
	float alpha;
};

static inline void ab_accum_body(void* ud, unsigned int begin, unsigned int end)
{
	const AbAccumCtx& c = *static_cast<const AbAccumCtx*>(ud);
	const unsigned int TILE_R = 64u;
	const unsigned int TILE_N = 64u;

	for (unsigned int m = begin; m < end; ++m)
	{
		float* crow = c.C + static_cast<size_t>(m) * c.N;
		for (unsigned int rb = 0; rb < c.R; rb += TILE_R)
		{
			const unsigned int re = (rb + TILE_R < c.R) ? (rb + TILE_R) : c.R;
			for (unsigned int nb = 0; nb < c.N; nb += TILE_N)
			{
				const unsigned int ne = (nb + TILE_N < c.N) ? (nb + TILE_N) : c.N;
				const unsigned int nlen = ne - nb;
				for (unsigned int r2 = rb; r2 < re; ++r2)
				{
					const float a = c.alpha * c.A[static_cast<size_t>(m) * c.R + r2];
					axpy_f32(crow + nb, &c.B[static_cast<size_t>(r2) * c.N + nb], a, nlen);
				}
			}
		}
	}
}

struct AbtCtx
{
	float* C;
	const float* A;
	const float* B;
	unsigned int M, K, N;
	float alpha;
};

static inline void abt_body(void* ud, unsigned int begin, unsigned int end)
{
	const AbtCtx& c = *static_cast<const AbtCtx*>(ud);
	const unsigned int TILE_K = 64u;
	const unsigned int TILE_N = 64u;

	for (unsigned int m = begin; m < end; ++m)
	{
		float* crow = c.C + static_cast<size_t>(m) * c.N;
		std::memset(crow, 0, static_cast<size_t>(c.N) * sizeof(float));
		for (unsigned int nb = 0; nb < c.N; nb += TILE_N)
		{
			const unsigned int ne = (nb + TILE_N < c.N) ? (nb + TILE_N) : c.N;
			for (unsigned int kb = 0; kb < c.K; kb += TILE_K)
			{
				const unsigned int ke = (kb + TILE_K < c.K) ? (kb + TILE_K) : c.K;
				const unsigned int klen = ke - kb;
				for (unsigned int n = nb; n < ne; ++n)
				{
					const float d = dot_f32(
						&c.A[static_cast<size_t>(m) * c.K + kb],
						&c.B[static_cast<size_t>(n) * c.K + kb],
						klen);
					crow[n] += c.alpha * d;
				}
			}
		}
	}
}

// ---- Public API ----

// C[M,N] = alpha * A^T[M,K] * B[K,N]
// A is [K,M] row-major (so A^T is [M,K]), B is [K,N] row-major, C is [M,N] row-major.
// C is zeroed before accumulation.
inline void atb(float* GLADES_RESTRICT C,
                const float* GLADES_RESTRICT A, const float* GLADES_RESTRICT B,
                unsigned int M, unsigned int K, unsigned int N,
                float alpha)
{
#ifdef GLADES_HAVE_CBLAS
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            (int)M, (int)N, (int)K, alpha,
	            A, (int)M, B, (int)N, 0.0f, C, (int)N);
	return;
#endif
	const unsigned long long flops = static_cast<unsigned long long>(M) * K * N;
	glades::ThreadPool& pool = glades::ThreadPool::instance();
	if (flops >= kParallelThreshold && M > 1u && pool.numThreads() > 1u)
	{
		AtbCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.K = K; ctx.N = N;
		ctx.alpha = alpha;
		pool.parallel_for(M, atb_body, &ctx);
	}
	else
	{
		AtbCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.K = K; ctx.N = N;
		ctx.alpha = alpha;
		atb_body(&ctx, 0, M);
	}
}

// C[M,N] += alpha * A[M,R] * B[R,N]
// A is [M,R] row-major, B is [R,N] row-major, C is [M,N] row-major.
// C is NOT zeroed — results are accumulated into existing values.
inline void ab_accum(float* GLADES_RESTRICT C,
                     const float* GLADES_RESTRICT A, const float* GLADES_RESTRICT B,
                     unsigned int M, unsigned int R, unsigned int N,
                     float alpha)
{
#ifdef GLADES_HAVE_CBLAS
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            (int)M, (int)N, (int)R, alpha,
	            A, (int)R, B, (int)N, 1.0f, C, (int)N);
	return;
#endif
	const unsigned long long flops = static_cast<unsigned long long>(M) * R * N;
	glades::ThreadPool& pool = glades::ThreadPool::instance();
	if (flops >= kParallelThreshold && M > 1u && pool.numThreads() > 1u)
	{
		AbAccumCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.R = R; ctx.N = N;
		ctx.alpha = alpha;
		pool.parallel_for(M, ab_accum_body, &ctx);
	}
	else
	{
		AbAccumCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.R = R; ctx.N = N;
		ctx.alpha = alpha;
		ab_accum_body(&ctx, 0, M);
	}
}

// C[M,N] = alpha * A[M,K] * B^T[K,N]
// A is [M,K] row-major, B is [N,K] row-major (so B^T is [K,N]), C is [M,N] row-major.
// C is zeroed before accumulation.
inline void abt(float* GLADES_RESTRICT C,
                const float* GLADES_RESTRICT A, const float* GLADES_RESTRICT B,
                unsigned int M, unsigned int K, unsigned int N,
                float alpha)
{
#ifdef GLADES_HAVE_CBLAS
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
	            (int)M, (int)N, (int)K, alpha,
	            A, (int)K, B, (int)K, 0.0f, C, (int)N);
	return;
#endif
	const unsigned long long flops = static_cast<unsigned long long>(M) * K * N;
	glades::ThreadPool& pool = glades::ThreadPool::instance();
	if (flops >= kParallelThreshold && M > 1u && pool.numThreads() > 1u)
	{
		AbtCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.K = K; ctx.N = N;
		ctx.alpha = alpha;
		pool.parallel_for(M, abt_body, &ctx);
	}
	else
	{
		AbtCtx ctx;
		ctx.C = C; ctx.A = A; ctx.B = B;
		ctx.M = M; ctx.K = K; ctx.N = N;
		ctx.alpha = alpha;
		abt_body(&ctx, 0, M);
	}
}

} // namespace gemm
} // namespace glades
