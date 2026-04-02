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
#ifndef GLADES_FFT_H
#define GLADES_FFT_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades
{

// A minimal complex number for C++98 (avoids <complex> ABI issues with some compilers)
struct Complex
{
	double re;
	double im;

	Complex() : re(0.0), im(0.0) {}
	Complex(double r, double i) : re(r), im(i) {}

	Complex operator+(const Complex& o) const { return Complex(re + o.re, im + o.im); }
	Complex operator-(const Complex& o) const { return Complex(re - o.re, im - o.im); }
	Complex operator*(const Complex& o) const
	{
		return Complex(re * o.re - im * o.im, re * o.im + im * o.re);
	}
	Complex conj() const { return Complex(re, -im); }
	double mag2() const { return re * re + im * im; }
	double mag() const { return std::sqrt(mag2()); }
};

class FFT
{
public:

	// Forward DFT of a real-valued signal.
	// Returns N complex coefficients where N = nextPowerOfTwo(signal.size()).
	// The input is zero-padded if its length is not a power of two.
	static std::vector<Complex> forward(const std::vector<double>& signal);

	// Inverse DFT. Returns a real-valued signal of length n (complex parts discarded).
	// If n == 0, uses coeffs.size().
	static std::vector<double> inverse(const std::vector<Complex>& coeffs, size_t n = 0);

	// Power spectral density: |X[k]|^2 / N for k = 0..N/2
	// Returns N/2 + 1 values (DC through Nyquist).
	static std::vector<double> powerSpectrum(const std::vector<double>& signal);

	// Cross-spectral density between two real signals of equal length.
	// Returns N/2 + 1 complex values: X[k] * conj(Y[k]) / N
	static std::vector<Complex> crossSpectrum(const std::vector<double>& x,
	                                          const std::vector<double>& y);

	// Apply a Hanning window in-place: w[t] = 0.5 * (1 - cos(2*pi*t / (N-1)))
	static void hanningWindow(std::vector<double>& signal);

	// Band-pass filter: zero out all frequencies outside [freqLo, freqHi] (in bins),
	// then inverse-FFT back to time domain.
	static std::vector<double> bandPassFilter(const std::vector<double>& signal,
	                                          size_t freqLo, size_t freqHi);

	// Compute the cross-spectral density matrix for N assets over T observations.
	// data: N x T (each row is one asset's return series)
	// Returns a flat vector of size (N/2+1) * N * N * 2, representing for each frequency
	// bin k the Hermitian matrix S(k) stored as interleaved [re, im] in row-major order.
	// Access element (i,j) at freq k: index = k * (N_assets*N_assets*2) + i*N_assets*2 + j*2
	static std::vector<double> crossSpectralMatrix(
		const std::vector<std::vector<double> >& data);

	// Integrate the cross-spectral matrix over a frequency band [freqLo, freqHi].
	// Returns a flat N*N real matrix (the real part of the band-integrated spectral density).
	// csdMatrix: output of crossSpectralMatrix()
	// numAssets: N
	// numFreqBins: number of frequency bins (T_padded/2 + 1)
	static std::vector<double> bandCovariance(const std::vector<double>& csdMatrix,
	                                          size_t numAssets,
	                                          size_t numFreqBins,
	                                          size_t freqLo,
	                                          size_t freqHi);

	// Next power of two >= n
	static size_t nextPow2(size_t n);

private:

	// In-place Cooley-Tukey radix-2 DIT FFT
	static void fftInPlace(std::vector<Complex>& x, bool inverse);

	// Bit-reversal permutation
	static void bitReverse(std::vector<Complex>& x);
};

}; // namespace glades

#endif
