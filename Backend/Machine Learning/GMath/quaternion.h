// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _QUATERNION_H
#define _QUATERNION_H

#include <cmath>
#include <limits>

namespace glades {

// 16-byte contiguous quaternion q = r + i*I + j*J + k*K in H.
// C++98-compatible header-only primitive for Quaternion Neural Networks (TYPE_QNN).
struct Quaternion
{
	float r;
	float i;
	float j;
	float k;

	Quaternion() : r(0.0f), i(0.0f), j(0.0f), k(0.0f) {}
	Quaternion(float r_, float i_, float j_, float k_) : r(r_), i(i_), j(j_), k(k_) {}

	static Quaternion zero()
	{
		return Quaternion(0.0f, 0.0f, 0.0f, 0.0f);
	}

	static Quaternion one()
	{
		return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
	}

	Quaternion conj() const
	{
		return Quaternion(r, -i, -j, -k);
	}

	float normSq() const
	{
		return (r * r) + (i * i) + (j * j) + (k * k);
	}

	float norm() const
	{
		return sqrtf(normSq());
	}

	Quaternion normalized(float eps = 1e-12f) const
	{
		const float n = norm();
		if (n <= eps)
			return zero();
		const float inv = 1.0f / n;
		return Quaternion(r * inv, i * inv, j * inv, k * inv);
	}

	float component(unsigned int lane) const
	{
		switch (lane)
		{
		case 0u: return r;
		case 1u: return i;
		case 2u: return j;
		case 3u: return k;
		default: return 0.0f;
		}
	}

	void setComponent(unsigned int lane, float v)
	{
		switch (lane)
		{
		case 0u: r = v; break;
		case 1u: i = v; break;
		case 2u: j = v; break;
		case 3u: k = v; break;
		default: break;
		}
	}

	static bool isFiniteScalar(float v)
	{
#if defined(__cplusplus) && __cplusplus >= 201103L
		return std::isfinite(v);
#else
		const float inf = std::numeric_limits<float>::infinity();
		return (v == v) && (v != inf) && (v != -inf);
#endif
	}

	bool isFinite() const
	{
		return isFiniteScalar(r) && isFiniteScalar(i) && isFiniteScalar(j) && isFiniteScalar(k);
	}

	static float dot(const Quaternion& a, const Quaternion& b)
	{
		return (a.r * b.r) + (a.i * b.i) + (a.j * b.j) + (a.k * b.k);
	}

	// Non-commutative Hamilton product: a (x) b.
	static Quaternion mul(const Quaternion& a, const Quaternion& b)
	{
		return Quaternion(
		    (a.r * b.r) - (a.i * b.i) - (a.j * b.j) - (a.k * b.k),
		    (a.r * b.i) + (a.i * b.r) + (a.j * b.k) - (a.k * b.j),
		    (a.r * b.j) - (a.i * b.k) + (a.j * b.r) + (a.k * b.i),
		    (a.r * b.k) + (a.i * b.j) - (a.j * b.i) + (a.k * b.r));
	}

	// Right-conjugated Hamilton product: a (x) b*.
	// Used for weight gradients: dL/dW = delta (x) a*.
	static Quaternion mulConjRight(const Quaternion& a, const Quaternion& b)
	{
		return Quaternion(
		    (a.r * b.r) + (a.i * b.i) + (a.j * b.j) + (a.k * b.k),
		    -(a.r * b.i) + (a.i * b.r) - (a.j * b.k) + (a.k * b.j),
		    -(a.r * b.j) + (a.i * b.k) + (a.j * b.r) - (a.k * b.i),
		    -(a.r * b.k) - (a.i * b.j) + (a.j * b.i) + (a.k * b.r));
	}

	// Left-conjugated Hamilton product: a* (x) b.
	// Used for backpropagating deltas: dL/da = sum_m W_m* (x) delta_m.
	static Quaternion mulConjLeft(const Quaternion& a, const Quaternion& b)
	{
		return Quaternion(
		    (a.r * b.r) + (a.i * b.i) + (a.j * b.j) + (a.k * b.k),
		    (a.r * b.i) - (a.i * b.r) - (a.j * b.k) + (a.k * b.j),
		    (a.r * b.j) + (a.i * b.k) - (a.j * b.r) - (a.k * b.i),
		    (a.r * b.k) - (a.i * b.j) + (a.j * b.i) - (a.k * b.r));
	}

	Quaternion operator-() const
	{
		return Quaternion(-r, -i, -j, -k);
	}

	Quaternion operator+(const Quaternion& rhs) const
	{
		return Quaternion(r + rhs.r, i + rhs.i, j + rhs.j, k + rhs.k);
	}

	Quaternion operator-(const Quaternion& rhs) const
	{
		return Quaternion(r - rhs.r, i - rhs.i, j - rhs.j, k - rhs.k);
	}

	Quaternion operator*(const Quaternion& rhs) const
	{
		return mul(*this, rhs);
	}

	Quaternion operator*(float s) const
	{
		return Quaternion(r * s, i * s, j * s, k * s);
	}

	Quaternion operator/(float s) const
	{
		const float inv = 1.0f / s;
		return Quaternion(r * inv, i * inv, j * inv, k * inv);
	}

	Quaternion& operator+=(const Quaternion& rhs)
	{
		r += rhs.r;
		i += rhs.i;
		j += rhs.j;
		k += rhs.k;
		return *this;
	}

	Quaternion& operator-=(const Quaternion& rhs)
	{
		r -= rhs.r;
		i -= rhs.i;
		j -= rhs.j;
		k -= rhs.k;
		return *this;
	}

	Quaternion& operator*=(float s)
	{
		r *= s;
		i *= s;
		j *= s;
		k *= s;
		return *this;
	}

	Quaternion& operator/=(float s)
	{
		const float inv = 1.0f / s;
		r *= inv;
		i *= inv;
		j *= inv;
		k *= inv;
		return *this;
	}
};

inline Quaternion operator*(float s, const Quaternion& q)
{
	return Quaternion(s * q.r, s * q.i, s * q.j, s * q.k);
}

// Shared boundary layout modes for mapping real-valued features/outputs <-> H.
enum QuaternionInputPackingMode
{
	// Pack up to 4 real input features per quaternion's (r, i, j, k) lanes
	// with zero-padding for trailing lanes: Q_in = ceil(D_real / 4).
	QUAT_PACK_GROUPS_OF_4 = 0,
	// Embed each real input feature into the real lane of its own quaternion:
	// x_f -> (x_f, 0, 0, 0), Q_in = D_real.
	QUAT_ONE_PER_FEATURE = 1
};

enum QuaternionOutputReadoutMode
{
	// One output quaternion per scalar output (Q_out = D_out); read/write .r.
	QUAT_READOUT_REAL_PART = 0,
	// Pack up to 4 scalar outputs into each output quaternion's (r, i, j, k) lanes:
	// Q_out = ceil(D_out / 4).
	QUAT_READOUT_SPLIT_COMPONENTS = 1
};

inline unsigned int quaternionInputCount(unsigned int realFeatureCount, QuaternionInputPackingMode packing)
{
	if (realFeatureCount == 0u)
		return 0u;
	return (packing == QUAT_PACK_GROUPS_OF_4) ? ((realFeatureCount + 3u) / 4u) : realFeatureCount;
}

inline unsigned int quaternionOutputCount(unsigned int realOutputCount, QuaternionOutputReadoutMode readout)
{
	if (realOutputCount == 0u)
		return 0u;
	return (readout == QUAT_READOUT_SPLIT_COMPONENTS) ? ((realOutputCount + 3u) / 4u) : realOutputCount;
}

inline void encodeRealToQuaternions(const float* src,
                                    unsigned int srcSize,
                                    unsigned int realFeatureCount,
                                    QuaternionInputPackingMode packing,
                                    Quaternion* dst,
                                    unsigned int qIn)
{
	if (!dst)
		return;
	for (unsigned int q = 0u; q < qIn; ++q)
		dst[q] = Quaternion::zero();

	if (packing == QUAT_PACK_GROUPS_OF_4)
	{
		for (unsigned int f = 0u; f < realFeatureCount; ++f)
		{
			const unsigned int qIdx = f / 4u;
			if (qIdx >= qIn)
				break;
			const float x = (src && (f < srcSize)) ? src[f] : 0.0f;
			dst[qIdx].setComponent(f % 4u, x);
		}
	}
	else
	{
		for (unsigned int f = 0u; f < realFeatureCount && f < qIn; ++f)
		{
			const float x = (src && (f < srcSize)) ? src[f] : 0.0f;
			dst[f] = Quaternion(x, 0.0f, 0.0f, 0.0f);
		}
	}
}

inline void encodeTargetsToQuaternions(const float* src,
                                       unsigned int srcSize,
                                       unsigned int realOutCount,
                                       QuaternionOutputReadoutMode readout,
                                       Quaternion* dst,
                                       unsigned int qOut)
{
	if (!dst)
		return;
	for (unsigned int q = 0u; q < qOut; ++q)
		dst[q] = Quaternion::zero();

	if (readout == QUAT_READOUT_SPLIT_COMPONENTS)
	{
		for (unsigned int k = 0u; k < realOutCount; ++k)
		{
			const unsigned int qIdx = k / 4u;
			if (qIdx >= qOut)
				break;
			const float y = (src && (k < srcSize)) ? src[k] : 0.0f;
			dst[qIdx].setComponent(k % 4u, y);
		}
	}
	else
	{
		for (unsigned int k = 0u; k < realOutCount && k < qOut; ++k)
		{
			const float y = (src && (k < srcSize)) ? src[k] : 0.0f;
			dst[k] = Quaternion(y, 0.0f, 0.0f, 0.0f);
		}
	}
}

inline void decodeQuaternionsToReal(const Quaternion* src,
                                    unsigned int qOut,
                                    QuaternionOutputReadoutMode readout,
                                    float* dst,
                                    unsigned int realOutCount)
{
	if (!dst)
		return;
	if (readout == QUAT_READOUT_SPLIT_COMPONENTS)
	{
		for (unsigned int k = 0u; k < realOutCount; ++k)
		{
			const unsigned int qIdx = k / 4u;
			dst[k] = (src && qIdx < qOut) ? src[qIdx].component(k % 4u) : 0.0f;
		}
	}
	else
	{
		for (unsigned int k = 0u; k < realOutCount; ++k)
			dst[k] = (src && k < qOut) ? src[k].r : 0.0f;
	}
}

} // namespace glades

#endif
