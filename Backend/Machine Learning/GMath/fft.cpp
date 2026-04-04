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
#include "fft.h"
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace glades
{

size_t FFT::nextPow2(size_t n)
{
	if (n == 0)
		return 1;
	size_t p = 1;
	while (p < n)
		p <<= 1;
	return p;
}

void FFT::bitReverse(std::vector<Complex>& x)
{
	size_t N = x.size();
	size_t bits = 0;
	{
		size_t tmp = N;
		while (tmp > 1)
		{
			tmp >>= 1;
			++bits;
		}
	}
	for (size_t i = 0; i < N; ++i)
	{
		size_t j = 0;
		for (size_t b = 0; b < bits; ++b)
		{
			if (i & ((size_t)1 << b))
				j |= ((size_t)1 << (bits - 1 - b));
		}
		if (i < j)
		{
			Complex tmp = x[i];
			x[i] = x[j];
			x[j] = tmp;
		}
	}
}

void FFT::fftInPlace(std::vector<Complex>& x, bool inv)
{
	size_t N = x.size();
	if (N <= 1)
		return;

	bitReverse(x);

	// Cooley-Tukey butterfly passes
	for (size_t len = 2; len <= N; len <<= 1)
	{
		double angle = 2.0 * M_PI / (double)len * (inv ? -1.0 : 1.0);
		Complex wn(std::cos(angle), std::sin(angle));
		for (size_t i = 0; i < N; i += len)
		{
			Complex w(1.0, 0.0);
			for (size_t j = 0; j < len / 2; ++j)
			{
				Complex u = x[i + j];
				Complex v = w * x[i + j + len / 2];
				x[i + j] = u + v;
				x[i + j + len / 2] = u - v;
				w = w * wn;
			}
		}
	}

	if (inv)
	{
		for (size_t i = 0; i < N; ++i)
		{
			x[i].re /= (double)N;
			x[i].im /= (double)N;
		}
	}
}

std::vector<Complex> FFT::forward(const std::vector<double>& signal)
{
	if (signal.empty())
		return std::vector<Complex>();

	size_t N = nextPow2(signal.size());
	std::vector<Complex> x(N);
	for (size_t i = 0; i < signal.size(); ++i)
	{
		x[i].re = signal[i];
		x[i].im = 0.0;
	}
	// Zero-pad remainder is already handled by Complex default constructor

	fftInPlace(x, false);
	return x;
}

std::vector<double> FFT::inverse(const std::vector<Complex>& coeffs, size_t n)
{
	if (coeffs.empty())
		return std::vector<double>();

	std::vector<Complex> x = coeffs;
	// Pad to power of two if needed
	size_t N = nextPow2(x.size());
	x.resize(N);

	fftInPlace(x, true);

	size_t outLen = (n > 0) ? n : N;
	if (outLen > N)
		outLen = N;
	std::vector<double> result(outLen);
	for (size_t i = 0; i < outLen; ++i)
		result[i] = x[i].re;
	return result;
}

std::vector<double> FFT::powerSpectrum(const std::vector<double>& signal)
{
	std::vector<Complex> X = forward(signal);
	size_t N = X.size();
	size_t numBins = N / 2 + 1;
	double scale = 1.0 / (double)N;

	std::vector<double> psd(numBins);
	for (size_t k = 0; k < numBins; ++k)
		psd[k] = X[k].mag2() * scale;
	return psd;
}

std::vector<Complex> FFT::crossSpectrum(const std::vector<double>& x,
                                        const std::vector<double>& y)
{
	std::vector<Complex> X = forward(x);
	std::vector<Complex> Y = forward(y);
	size_t N = X.size();
	size_t numBins = N / 2 + 1;
	double scale = 1.0 / (double)N;

	std::vector<Complex> csd(numBins);
	for (size_t k = 0; k < numBins; ++k)
	{
		// S_xy(k) = X(k) * conj(Y(k)) / N
		Complex Yc = Y[k].conj();
		csd[k].re = (X[k].re * Yc.re - X[k].im * Yc.im) * scale;
		csd[k].im = (X[k].re * Yc.im + X[k].im * Yc.re) * scale;
	}
	return csd;
}

void FFT::hanningWindow(std::vector<double>& signal)
{
	size_t N = signal.size();
	if (N <= 1)
		return;
	for (size_t t = 0; t < N; ++t)
	{
		double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * (double)t / (double)(N - 1)));
		signal[t] *= w;
	}
}

std::vector<double> FFT::bandPassFilter(const std::vector<double>& signal,
                                        size_t freqLo, size_t freqHi)
{
	std::vector<Complex> X = forward(signal);
	size_t N = X.size();

	// Zero out frequencies outside the band
	for (size_t k = 0; k < N; ++k)
	{
		// Map k to the corresponding positive frequency bin
		size_t posBin = k;
		if (k > N / 2)
			posBin = N - k;

		if (posBin < freqLo || posBin > freqHi)
		{
			X[k].re = 0.0;
			X[k].im = 0.0;
		}
	}

	return inverse(X, signal.size());
}

std::vector<double> FFT::crossSpectralMatrix(
	const std::vector<std::vector<double> >& data)
{
	if (data.empty())
		return std::vector<double>();

	size_t numAssets = data.size();
	size_t T = data[0].size();
	size_t N = nextPow2(T);
	size_t numBins = N / 2 + 1;

	// Compute FFT for each asset
	std::vector<std::vector<Complex> > spectra(numAssets);
	for (size_t i = 0; i < numAssets; ++i)
		spectra[i] = forward(data[i]);

	// Build cross-spectral density matrix for each frequency bin
	// Layout: for each bin k, a flat N_assets x N_assets matrix of Complex
	// Stored as interleaved [re, im]
	size_t matSize = numBins * numAssets * numAssets * 2;
	std::vector<double> csdMatrix(matSize, 0.0);
	double scale = 1.0 / (double)N;

	for (size_t k = 0; k < numBins; ++k)
	{
		size_t baseIdx = k * (numAssets * numAssets * 2);
		for (size_t i = 0; i < numAssets; ++i)
		{
			for (size_t j = 0; j < numAssets; ++j)
			{
				// S_ij(k) = X_i(k) * conj(X_j(k)) / N
				Complex Xi = spectra[i][k];
				Complex Xjc = spectra[j][k].conj();
				double re = (Xi.re * Xjc.re - Xi.im * Xjc.im) * scale;
				double im = (Xi.re * Xjc.im + Xi.im * Xjc.re) * scale;
				size_t idx = baseIdx + i * numAssets * 2 + j * 2;
				csdMatrix[idx] = re;
				csdMatrix[idx + 1] = im;
			}
		}
	}
	return csdMatrix;
}

std::vector<double> FFT::bandCovariance(const std::vector<double>& csdMatrix,
                                        size_t numAssets,
                                        size_t numFreqBins,
                                        size_t freqLo,
                                        size_t freqHi)
{
	size_t matN = numAssets * numAssets;
	std::vector<double> bandCov(matN, 0.0);

	if (freqHi >= numFreqBins)
		freqHi = numFreqBins - 1;
	if (freqLo > freqHi)
		return bandCov;

	// Integrate the real part of S(k) over the band, scaled by 2*pi/N_padded
	// The factor 2*pi cancels with the 1/(2*pi) in the spectral density definition.
	// We multiply by 2 for k > 0 and k < N/2 to account for negative frequencies.
	for (size_t k = freqLo; k <= freqHi; ++k)
	{
		size_t baseIdx = k * (numAssets * numAssets * 2);
		double freqWeight = 1.0;
		if (k > 0 && k < numFreqBins - 1)
			freqWeight = 2.0; // Double for symmetric negative frequencies

		for (size_t i = 0; i < numAssets; ++i)
		{
			for (size_t j = 0; j < numAssets; ++j)
			{
				double re = csdMatrix[baseIdx + i * numAssets * 2 + j * 2];
				bandCov[i * numAssets + j] += re * freqWeight;
			}
		}
	}

	return bandCov;
}

}; // namespace glades
