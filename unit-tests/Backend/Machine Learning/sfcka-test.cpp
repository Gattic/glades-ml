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
#include "sfcka-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/GMath/fft.h"
#include "../../../Backend/Machine Learning/GMath/fisher_transform.h"
#include "../../../Backend/Machine Learning/GMath/kelly.h"
#include "../../../Backend/Machine Learning/GMath/qp_solver.h"
#include "Backend/Plotter/Plotter.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

namespace
{

static const std::string IMAGE_DIR = "datasets";

// Simple LCG for deterministic pseudo-random numbers
struct LCG
{
	unsigned long state;
	LCG(unsigned long seed) : state(seed) {}
	unsigned long next()
	{
		state = state * 1103515245UL + 12345UL;
		return (state >> 16) & 0x7FFF;
	}
	double uniform()
	{
		return next() / 32768.0;
	}
	double normal()
	{
		double u1 = uniform();
		double u2 = uniform();
		if (u1 < 1e-10) u1 = 1e-10;
		return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
	}
};

bool approxEqual(double a, double b, double tol = 1e-6)
{
	return std::fabs(a - b) < tol;
}

// Save a line chart of 1D data (e.g. time-domain signal or spectrum)
void saveLineChart(const std::string& filename, const std::string& title,
                   const std::string& xLabel, const std::string& yLabel,
                   const std::string& seriesName,
                   const std::vector<double>& yData,
                   const shmea::RGBA& color)
{
	std::vector<shmea::Point> points;
	for (size_t i = 0; i < yData.size(); ++i)
	{
		shmea::Point p;
		p.x = (float)i;
		p.y = (float)yData[i];
		points.push_back(p);
	}

	shmea::Plotter plotter(1800, 1000, 1);
	plotter.chart()
		.title(title, 36)
		.grid(true)
		.axes(true)
		.cornerRadius(15)
		.logo("logo.png")
		.axisLabels(xLabel, yLabel, 28)
		.autoMargins(shmea::CHART_LINE)
		.addSeries(seriesName, points, color, shmea::SERIES_LINE, 2, 8)
		.saveAs(filename, IMAGE_DIR);

	printf("[SFCKA] Image saved: %s/%s\n", IMAGE_DIR.c_str(), filename.c_str());
}

// Save a scatter chart of 2D points
void saveScatterChart(const std::string& filename, const std::string& title,
                      const std::string& xLabel, const std::string& yLabel,
                      const std::string& seriesName,
                      const std::vector<shmea::Point>& points,
                      const shmea::RGBA& color)
{
	shmea::Plotter plotter(1800, 1000, 1);
	plotter.chart()
		.title(title, 36)
		.grid(true)
		.axes(true)
		.cornerRadius(15)
		.logo("logo.png")
		.axisLabels(xLabel, yLabel, 28)
		.autoMargins(shmea::CHART_SCATTER)
		.addSeries(seriesName, points, color, shmea::SERIES_SCATTER, 2, 6)
		.saveAs(filename, IMAGE_DIR);

	printf("[SFCKA] Image saved: %s/%s\n", IMAGE_DIR.c_str(), filename.c_str());
}

// Save a multi-series line chart
void saveMultiLineChart(const std::string& filename, const std::string& title,
                        const std::string& xLabel, const std::string& yLabel,
                        const std::vector<std::string>& names,
                        const std::vector<std::vector<double> >& series,
                        const std::vector<shmea::RGBA>& colors)
{
	shmea::Plotter plotter(1800, 1000, 1);
	shmea::ChartBuilder builder = plotter.chart()
		.title(title, 36)
		.grid(true)
		.axes(true)
		.cornerRadius(15)
		.logo("logo.png")
		.axisLabels(xLabel, yLabel, 28)
		.autoMargins(shmea::CHART_LINE);

	for (size_t s = 0; s < series.size(); ++s)
	{
		std::vector<shmea::Point> points;
		for (size_t i = 0; i < series[s].size(); ++i)
		{
			shmea::Point p;
			p.x = (float)i;
			p.y = (float)series[s][i];
			points.push_back(p);
		}
		builder.addSeries(names[s], points, colors[s], shmea::SERIES_LINE, 2, 8);
	}

	builder.saveAs(filename, IMAGE_DIR);
	printf("[SFCKA] Image saved: %s/%s\n", IMAGE_DIR.c_str(), filename.c_str());
}

} // anonymous namespace

// ============================================================================
// FFT UNIT TESTS
// ============================================================================

void FFTUnitTest(bool saveImages)
{
	printf("============================================================\n");
	printf("  FFT Unit Tests\n");
	printf("============================================================\n");

	// ---- Test 1: nextPow2 ----
	printf("\n--- Test 1: nextPow2 ---\n");
	ASSERT("nextPow2(1) == 1", glades::FFT::nextPow2(1) == 1);
	ASSERT("nextPow2(2) == 2", glades::FFT::nextPow2(2) == 2);
	ASSERT("nextPow2(3) == 4", glades::FFT::nextPow2(3) == 4);
	ASSERT("nextPow2(5) == 8", glades::FFT::nextPow2(5) == 8);
	ASSERT("nextPow2(128) == 128", glades::FFT::nextPow2(128) == 128);
	ASSERT("nextPow2(129) == 256", glades::FFT::nextPow2(129) == 256);

	// ---- Test 2: Forward FFT of a DC signal ----
	printf("\n--- Test 2: DC signal FFT ---\n");
	{
		size_t N = 8;
		std::vector<double> signal(N, 3.0);
		std::vector<glades::Complex> X = glades::FFT::forward(signal);
		ASSERT("FFT output size", X.size() == N);
		// DC component should be N * 3.0 = 24
		ASSERT("DC component real", approxEqual(X[0].re, 24.0, 1e-8));
		ASSERT("DC component imag", approxEqual(X[0].im, 0.0, 1e-8));
		// All other components should be 0
		for (size_t k = 1; k < N; ++k)
		{
			ASSERT("non-DC real ~0", approxEqual(X[k].re, 0.0, 1e-8));
			ASSERT("non-DC imag ~0", approxEqual(X[k].im, 0.0, 1e-8));
		}
	}

	// ---- Test 3: Forward FFT of a pure cosine ----
	printf("\n--- Test 3: Pure cosine FFT ---\n");
	{
		size_t N = 64;
		size_t freqBin = 4; // 4 cycles in the window
		std::vector<double> signal(N);
		for (size_t t = 0; t < N; ++t)
			signal[t] = std::cos(2.0 * M_PI * (double)freqBin * (double)t / (double)N);

		std::vector<glades::Complex> X = glades::FFT::forward(signal);
		ASSERT("FFT output size", X.size() == N);

		// Peak should be at bin freqBin and N-freqBin
		double peakMag = X[freqBin].mag();
		ASSERT("peak magnitude", approxEqual(peakMag, (double)N / 2.0, 1e-6));

		// Other bins (except conjugate) should be near zero
		for (size_t k = 0; k < N; ++k)
		{
			if (k != freqBin && k != N - freqBin)
				ASSERT("non-peak bin near zero", X[k].mag() < 1e-8);
		}

		if (saveImages)
		{
			saveLineChart("fft_cosine_signal.png", "FFT: Pure Cosine Signal (4 cycles)",
			              "Sample", "Amplitude", "cos(2pi*4*t/64)", signal,
			              shmea::RGBA(0x00, 0x88, 0xFF, 0xFF));

			// Save magnitude spectrum
			std::vector<double> magSpectrum(N / 2 + 1);
			for (size_t k = 0; k <= N / 2; ++k)
				magSpectrum[k] = X[k].mag();
			saveLineChart("fft_cosine_magnitude.png",
			              "FFT: Magnitude Spectrum (peak at bin 4)",
			              "Frequency Bin", "Magnitude", "|X[k]|", magSpectrum,
			              shmea::RGBA(0xFF, 0x44, 0x00, 0xFF));
		}
	}

	// ---- Test 4: Inverse FFT recovers original ----
	printf("\n--- Test 4: Inverse FFT round-trip ---\n");
	{
		LCG rng(42);
		size_t N = 32;
		std::vector<double> original(N);
		for (size_t i = 0; i < N; ++i)
			original[i] = rng.normal();

		std::vector<glades::Complex> X = glades::FFT::forward(original);
		std::vector<double> recovered = glades::FFT::inverse(X, N);
		ASSERT("recovered size", recovered.size() == N);

		for (size_t i = 0; i < N; ++i)
			ASSERT("round-trip accuracy", approxEqual(original[i], recovered[i], 1e-8));
	}

	// ---- Test 5: Power spectrum ----
	printf("\n--- Test 5: Power spectrum ---\n");
	{
		size_t N = 128;
		size_t freqBin = 10;
		std::vector<double> signal(N);
		for (size_t t = 0; t < N; ++t)
			signal[t] = 2.0 * std::cos(2.0 * M_PI * (double)freqBin * (double)t / (double)N);

		std::vector<double> psd = glades::FFT::powerSpectrum(signal);
		ASSERT("PSD size", psd.size() == N / 2 + 1);

		// Find peak
		size_t maxIdx = 0;
		double maxVal = 0.0;
		for (size_t k = 0; k < psd.size(); ++k)
		{
			if (psd[k] > maxVal)
			{
				maxVal = psd[k];
				maxIdx = k;
			}
		}
		ASSERT("PSD peak at correct frequency", maxIdx == freqBin);

		if (saveImages)
		{
			saveLineChart("fft_power_spectrum.png",
			              "FFT: Power Spectral Density (peak at bin 10)",
			              "Frequency Bin", "Power", "PSD", psd,
			              shmea::RGBA(0x88, 0x00, 0xFF, 0xFF));
		}
	}

	// ---- Test 6: Parseval's theorem ----
	printf("\n--- Test 6: Parseval's theorem ---\n");
	{
		LCG rng(123);
		size_t N = 64;
		std::vector<double> signal(N);
		double timeDomainEnergy = 0.0;
		for (size_t i = 0; i < N; ++i)
		{
			signal[i] = rng.normal();
			timeDomainEnergy += signal[i] * signal[i];
		}

		std::vector<glades::Complex> X = glades::FFT::forward(signal);
		double freqDomainEnergy = 0.0;
		for (size_t k = 0; k < X.size(); ++k)
			freqDomainEnergy += X[k].mag2();
		freqDomainEnergy /= (double)N;

		ASSERT("Parseval's theorem", approxEqual(timeDomainEnergy, freqDomainEnergy, 1e-6));
	}

	// ---- Test 7: Hanning window ----
	printf("\n--- Test 7: Hanning window ---\n");
	{
		size_t N = 8;
		std::vector<double> signal(N, 1.0);
		glades::FFT::hanningWindow(signal);
		// Endpoints should be 0, middle should be 1
		ASSERT("Hanning endpoint 0", approxEqual(signal[0], 0.0, 1e-10));
		ASSERT("Hanning endpoint N-1", approxEqual(signal[N - 1], 0.0, 1e-10));
		// Midpoint: 0.5 * (1 - cos(2*pi*4/7)) = 0.5 * (1 - cos(8*pi/7))
		// For N=8, t=4: 0.5 * (1 - cos(8*pi/7))
		ASSERT("Hanning mid > 0", signal[N / 2] > 0.5);
	}

	// ---- Test 8: Band-pass filter ----
	printf("\n--- Test 8: Band-pass filter ---\n");
	{
		size_t N = 128;
		// Signal = low freq (2 cycles) + high freq (30 cycles)
		std::vector<double> signal(N);
		for (size_t t = 0; t < N; ++t)
		{
			signal[t] = std::cos(2.0 * M_PI * 2.0 * (double)t / (double)N) +
			            std::cos(2.0 * M_PI * 30.0 * (double)t / (double)N);
		}

		// Filter to keep only low frequencies (bins 0 to 5)
		std::vector<double> filtered = glades::FFT::bandPassFilter(signal, 0, 5);
		ASSERT("filtered size", filtered.size() == N);

		// Filtered should contain mostly the 2-cycle component
		// Check correlation with the 2-cycle cosine
		double corrLow = 0.0;
		double corrHigh = 0.0;
		for (size_t t = 0; t < N; ++t)
		{
			double low = std::cos(2.0 * M_PI * 2.0 * (double)t / (double)N);
			double high = std::cos(2.0 * M_PI * 30.0 * (double)t / (double)N);
			corrLow += filtered[t] * low;
			corrHigh += filtered[t] * high;
		}
		ASSERT("low freq preserved", std::fabs(corrLow) > std::fabs(corrHigh) * 10.0);

		if (saveImages)
		{
			std::vector<std::string> names;
			names.push_back("Original (low+high)");
			names.push_back("Band-pass filtered (low only)");

			std::vector<std::vector<double> > series;
			series.push_back(signal);
			series.push_back(filtered);

			std::vector<shmea::RGBA> colors;
			colors.push_back(shmea::RGBA(0xCC, 0xCC, 0xCC, 0xFF));
			colors.push_back(shmea::RGBA(0x00, 0xCC, 0x44, 0xFF));

			saveMultiLineChart("fft_bandpass.png",
			                   "FFT: Band-Pass Filter (low freq preserved)",
			                   "Sample", "Amplitude", names, series, colors);
		}
	}

	// ---- Test 9: Cross-spectral matrix and band covariance ----
	printf("\n--- Test 9: Cross-spectral matrix ---\n");
	{
		// Two perfectly correlated assets at one frequency
		size_t T = 64;
		size_t N = 2;
		std::vector<std::vector<double> > data(N);
		for (size_t i = 0; i < N; ++i)
			data[i].resize(T);

		for (size_t t = 0; t < T; ++t)
		{
			data[0][t] = std::cos(2.0 * M_PI * 3.0 * (double)t / (double)T);
			data[1][t] = 0.5 * std::cos(2.0 * M_PI * 3.0 * (double)t / (double)T);
		}

		std::vector<double> csd = glades::FFT::crossSpectralMatrix(data);
		ASSERT("CSD matrix non-empty", !csd.empty());

		size_t Npad = glades::FFT::nextPow2(T);
		size_t numBins = Npad / 2 + 1;

		// Band covariance over all frequencies
		std::vector<double> bandCov = glades::FFT::bandCovariance(csd, N, numBins, 0,
		                                                          numBins - 1);
		ASSERT("band cov size", bandCov.size() == N * N);

		// Asset 0 should have higher variance than asset 1
		ASSERT("asset 0 var > asset 1 var", bandCov[0] > bandCov[3]);
		// Cross-covariance should be positive (same signal, just scaled)
		ASSERT("positive cross-covariance", bandCov[1] > 0.0);
	}

	// ---- Test 10: Empty input handling ----
	printf("\n--- Test 10: Empty input ---\n");
	{
		std::vector<double> empty;
		std::vector<glades::Complex> X = glades::FFT::forward(empty);
		ASSERT("empty forward", X.empty());

		std::vector<glades::Complex> emptyC;
		std::vector<double> inv = glades::FFT::inverse(emptyC);
		ASSERT("empty inverse", inv.empty());
	}

	printf("\n  FFT: All tests passed.\n\n");
}

// ============================================================================
// FISHER TRANSFORM UNIT TESTS
// ============================================================================

void FisherTransformUnitTest(bool saveImages)
{
	printf("============================================================\n");
	printf("  Fisher Transform Unit Tests\n");
	printf("============================================================\n");

	// ---- Test 1: Basic transform values ----
	printf("\n--- Test 1: Basic transform values ---\n");
	{
		// arctanh(0) = 0
		ASSERT("transform(0) == 0", approxEqual(glades::FisherTransform::transform(0.0), 0.0));

		// arctanh(0.5) = 0.5493...
		double z05 = glades::FisherTransform::transform(0.5);
		ASSERT("transform(0.5) ~ 0.5493", approxEqual(z05, 0.5493061, 1e-5));

		// arctanh(-0.5) = -0.5493...
		double zm05 = glades::FisherTransform::transform(-0.5);
		ASSERT("transform(-0.5) ~ -0.5493", approxEqual(zm05, -0.5493061, 1e-5));

		// arctanh(0.9) = 1.4722...
		double z09 = glades::FisherTransform::transform(0.9);
		ASSERT("transform(0.9) ~ 1.4722", approxEqual(z09, 1.4722195, 1e-5));

		// arctanh(0.99) = 2.6466...
		double z099 = glades::FisherTransform::transform(0.99);
		ASSERT("transform(0.99) ~ 2.6467", approxEqual(z099, 2.6466524, 1e-4));
	}

	// ---- Test 2: Inverse transform ----
	printf("\n--- Test 2: Inverse transform ---\n");
	{
		ASSERT("inverse(0) == 0", approxEqual(glades::FisherTransform::inverse(0.0), 0.0));
		ASSERT("inverse(1) ~ 0.7616", approxEqual(glades::FisherTransform::inverse(1.0),
		                                           0.7615942, 1e-5));
		ASSERT("inverse(-1) ~ -0.7616", approxEqual(glades::FisherTransform::inverse(-1.0),
		                                            -0.7615942, 1e-5));
	}

	// ---- Test 3: Round-trip ----
	printf("\n--- Test 3: Round-trip ---\n");
	{
		double rhos[] = {-0.99, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.99};
		for (size_t i = 0; i < 9; ++i)
		{
			double z = glades::FisherTransform::transform(rhos[i]);
			double recovered = glades::FisherTransform::inverse(z);
			ASSERT("round-trip", approxEqual(recovered, rhos[i], 1e-8));
		}
	}

	// ---- Test 4: Clamping at boundaries ----
	printf("\n--- Test 4: Clamping at boundaries ---\n");
	{
		// Should not crash or return inf
		double z1 = glades::FisherTransform::transform(1.0);
		ASSERT("transform(1.0) is finite", z1 < 1e15 && z1 > -1e15);

		double zm1 = glades::FisherTransform::transform(-1.0);
		ASSERT("transform(-1.0) is finite", zm1 < 1e15 && zm1 > -1e15);

		double z2 = glades::FisherTransform::transform(1.5); // Beyond range
		ASSERT("transform(1.5) clamped and finite", z2 < 1e15);
	}

	// ---- Image: Fisher transform curve and shrinkage function ----
	if (saveImages)
	{
		// Plot arctanh(rho) for rho in [-0.99, 0.99]
		size_t nPts = 200;
		std::vector<shmea::Point> transformPts;
		for (size_t i = 0; i < nPts; ++i)
		{
			double rho = -0.99 + 1.98 * (double)i / (double)(nPts - 1);
			double z = glades::FisherTransform::transform(rho);
			shmea::Point p;
			p.x = (float)rho;
			p.y = (float)z;
			transformPts.push_back(p);
		}
		saveScatterChart("fisher_transform_curve.png",
		                 "Fisher Transform: z = arctanh(rho)",
		                 "rho (correlation)", "z (Fisher-transformed)",
		                 "arctanh(rho)", transformPts,
		                 shmea::RGBA(0x00, 0x66, 0xFF, 0xFF));

		// Plot shrinkage function psi(SNR) for SNR in [0, 5]
		std::vector<shmea::Point> shrinkPts;
		for (size_t i = 0; i < nPts; ++i)
		{
			double s = 5.0 * (double)i / (double)(nPts - 1);
			double psi = glades::FisherTransform::shrinkage(s);
			shmea::Point p;
			p.x = (float)s;
			p.y = (float)psi;
			shrinkPts.push_back(p);
		}
		saveScatterChart("fisher_shrinkage.png",
		                 "Fisher Shrinkage: psi(SNR) = 1 - exp(-SNR^2/2)",
		                 "Signal-to-Noise Ratio", "Shrinkage Weight",
		                 "psi(SNR)", shrinkPts,
		                 shmea::RGBA(0xFF, 0x44, 0x00, 0xFF));
	}

	// ---- Test 5: Vector transform ----
	printf("\n--- Test 5: Vector transform ---\n");
	{
		std::vector<double> rho;
		rho.push_back(-0.5);
		rho.push_back(0.0);
		rho.push_back(0.5);
		std::vector<double> z = glades::FisherTransform::transformVec(rho);
		ASSERT("vec transform size", z.size() == 3);
		ASSERT("vec[0] == transform(-0.5)",
		       approxEqual(z[0], glades::FisherTransform::transform(-0.5)));
		ASSERT("vec[1] == 0", approxEqual(z[1], 0.0));
		ASSERT("vec[2] == transform(0.5)",
		       approxEqual(z[2], glades::FisherTransform::transform(0.5)));
	}

	// ---- Test 6: SNR and shrinkage ----
	printf("\n--- Test 6: SNR and shrinkage ---\n");
	{
		// SNR with z=0.5 and n=100: 0.5 * sqrt(97) ~ 4.924
		double s = glades::FisherTransform::snr(0.5, 100);
		ASSERT("SNR value", approxEqual(s, 0.5 * std::sqrt(97.0), 1e-4));

		// Shrinkage at SNR=0: psi(0) = 1 - exp(0) = 0
		ASSERT("shrinkage(0) == 0", approxEqual(glades::FisherTransform::shrinkage(0.0), 0.0));

		// Shrinkage at large SNR: psi(5) ~ 1 - exp(-12.5) ~ 1
		ASSERT("shrinkage(5) ~ 1",
		       approxEqual(glades::FisherTransform::shrinkage(5.0), 1.0, 1e-4));

		// Shrinkage at SNR=1: psi(1) = 1 - exp(-0.5) ~ 0.3935
		ASSERT("shrinkage(1) ~ 0.3935",
		       approxEqual(glades::FisherTransform::shrinkage(1.0), 0.3934693, 1e-5));

		// Low sample size warning
		double v = glades::FisherTransform::variance(3);
		ASSERT("variance(3) is large", v > 1e5);
	}

	// ---- Test 7: Adjusted return ----
	printf("\n--- Test 7: Adjusted return ---\n");
	{
		// High SNR: adjusted return should be close to original
		double adj = glades::FisherTransform::adjustedReturn(0.01, 0.8, 200);
		ASSERT("high SNR preserves return", adj > 0.009);

		// Low SNR: adjusted return should be near zero
		double adjLow = glades::FisherTransform::adjustedReturn(0.01, 0.01, 10);
		ASSERT("low SNR shrinks return", adjLow < 0.005);
	}

	// ---- Test 8: Correlation matrix ----
	printf("\n--- Test 8: Correlation matrix ---\n");
	{
		// 3 features, 100 samples
		LCG rng(99);
		size_t M = 100;
		size_t N = 3;
		std::vector<std::vector<double> > data(M);
		for (size_t i = 0; i < M; ++i)
		{
			data[i].resize(N);
			double x = rng.normal();
			data[i][0] = x;
			data[i][1] = 0.8 * x + 0.2 * rng.normal(); // Correlated with [0]
			data[i][2] = rng.normal(); // Independent
		}

		std::vector<double> corr = glades::FisherTransform::correlationMatrix(data);
		ASSERT("corr matrix size", corr.size() == N * N);

		// Diagonal should be 1
		for (size_t i = 0; i < N; ++i)
			ASSERT("diagonal == 1", approxEqual(corr[i * N + i], 1.0, 1e-6));

		// Symmetry
		for (size_t i = 0; i < N; ++i)
			for (size_t j = 0; j < N; ++j)
				ASSERT("symmetry", approxEqual(corr[i * N + j], corr[j * N + i], 1e-10));

		// corr(0,1) should be high positive
		ASSERT("corr(0,1) > 0.5", corr[0 * N + 1] > 0.5);

		// corr(0,2) should be near zero
		ASSERT("|corr(0,2)| < 0.3", std::fabs(corr[0 * N + 2]) < 0.3);
	}

	// ---- Test 9: Covariance matrix ----
	printf("\n--- Test 9: Covariance matrix ---\n");
	{
		// Known data
		std::vector<std::vector<double> > data;
		std::vector<double> r1; r1.push_back(1.0); r1.push_back(2.0); data.push_back(r1);
		std::vector<double> r2; r2.push_back(3.0); r2.push_back(4.0); data.push_back(r2);
		std::vector<double> r3; r3.push_back(5.0); r3.push_back(6.0); data.push_back(r3);

		std::vector<double> cov = glades::FisherTransform::covarianceMatrix(data);
		ASSERT("cov size", cov.size() == 4);
		// Var(X1) = var([1,3,5]) = 4.0 (sample variance with N-1)
		ASSERT("var(X1) == 4", approxEqual(cov[0], 4.0, 1e-10));
		// Var(X2) = var([2,4,6]) = 4.0
		ASSERT("var(X2) == 4", approxEqual(cov[3], 4.0, 1e-10));
		// Cov(X1,X2) = 4.0 (perfectly correlated)
		ASSERT("cov(X1,X2) == 4", approxEqual(cov[1], 4.0, 1e-10));
	}

	// ---- Test 10: Column means and stddevs ----
	printf("\n--- Test 10: Column means and stddevs ---\n");
	{
		std::vector<std::vector<double> > data;
		std::vector<double> r1; r1.push_back(2.0); r1.push_back(4.0); data.push_back(r1);
		std::vector<double> r2; r2.push_back(4.0); r2.push_back(6.0); data.push_back(r2);
		std::vector<double> r3; r3.push_back(6.0); r3.push_back(8.0); data.push_back(r3);

		std::vector<double> means = glades::FisherTransform::columnMeans(data);
		ASSERT("mean[0] == 4", approxEqual(means[0], 4.0, 1e-10));
		ASSERT("mean[1] == 6", approxEqual(means[1], 6.0, 1e-10));

		std::vector<double> stds = glades::FisherTransform::columnStdDevs(data, means);
		ASSERT("std[0] == 2", approxEqual(stds[0], 2.0, 1e-10));
		ASSERT("std[1] == 2", approxEqual(stds[1], 2.0, 1e-10));
	}

	printf("\n  Fisher Transform: All tests passed.\n\n");
}

// ============================================================================
// KELLY CRITERION UNIT TESTS
// ============================================================================

void KellyUnitTest(bool saveImages)
{
	printf("============================================================\n");
	printf("  Kelly Criterion Unit Tests\n");
	printf("============================================================\n");

	// ---- Test 1: Univariate Kelly fraction ----
	printf("\n--- Test 1: Univariate Kelly fraction ---\n");
	{
		// mu = 0.01, sigma^2 = 0.04 => f* = 0.01/0.04 = 0.25
		double f = glades::Kelly::fraction(0.01, 0.04);
		ASSERT("Kelly fraction", approxEqual(f, 0.25, 1e-10));

		// Negative expected return: should bet negative (short)
		double fNeg = glades::Kelly::fraction(-0.02, 0.04);
		ASSERT("negative Kelly", approxEqual(fNeg, -0.5, 1e-10));

		// Zero variance: should return 0
		double fZero = glades::Kelly::fraction(0.01, 0.0);
		ASSERT("zero variance", approxEqual(fZero, 0.0, 1e-10));
	}

	// ---- Test 2: Univariate growth rate ----
	printf("\n--- Test 2: Univariate growth rate ---\n");
	{
		double mu = 0.01;
		double sigma2 = 0.04;
		double fOpt = glades::Kelly::fraction(mu, sigma2);
		double gOpt = glades::Kelly::growthRate(fOpt, mu, sigma2);
		double gExpected = glades::Kelly::optimalGrowthRate(mu, sigma2);

		// Optimal growth rate = mu^2 / (2*sigma^2) = 0.0001 / 0.08 = 0.00125
		ASSERT("optimal growth rate", approxEqual(gOpt, 0.00125, 1e-10));
		ASSERT("growth at optimum matches", approxEqual(gOpt, gExpected, 1e-10));

		// Sub-optimal fraction should have lower growth rate
		double gHalf = glades::Kelly::growthRate(fOpt * 0.5, mu, sigma2);
		ASSERT("sub-optimal growth lower", gHalf < gOpt);

		// Double Kelly should have lower growth rate (overbetting)
		double gDouble = glades::Kelly::growthRate(fOpt * 2.0, mu, sigma2);
		ASSERT("overbetting growth lower", gDouble < gOpt);
	}

	// ---- Image: Kelly growth rate as a function of fraction ----
	if (saveImages)
	{
		double mu = 0.01;
		double sigma2 = 0.04;
		double fOpt = glades::Kelly::fraction(mu, sigma2);
		size_t nPts = 200;
		std::vector<shmea::Point> growthPts;
		for (size_t i = 0; i < nPts; ++i)
		{
			double f = -0.5 + 1.5 * (double)i / (double)(nPts - 1);
			double g = glades::Kelly::growthRate(f, mu, sigma2);
			shmea::Point p;
			p.x = (float)f;
			p.y = (float)g;
			growthPts.push_back(p);
		}
		saveScatterChart("kelly_growth_curve.png",
		                 "Kelly: Growth Rate G(f) = f*mu - 0.5*f^2*sigma^2",
		                 "Fraction f", "Growth Rate G(f)",
		                 "G(f), mu=0.01, sig2=0.04", growthPts,
		                 shmea::RGBA(0x00, 0xAA, 0x44, 0xFF));

		// Mark the optimum
		printf("  [Kelly Image] Optimal f* = %.4f, G* = %.6f\n",
		       fOpt, glades::Kelly::optimalGrowthRate(mu, sigma2));
	}

	// ---- Test 3: Multivariate Kelly (uncorrelated factors) ----
	printf("\n--- Test 3: Factor fractions ---\n");
	{
		std::vector<double> alphas;
		alphas.push_back(0.02);
		alphas.push_back(0.01);
		alphas.push_back(-0.005);

		std::vector<double> vars;
		vars.push_back(0.04);
		vars.push_back(0.01);
		vars.push_back(0.02);

		std::vector<double> g = glades::Kelly::factorFractions(alphas, vars);
		ASSERT("factor fraction count", g.size() == 3);
		ASSERT("g[0] = 0.02/0.04 = 0.5", approxEqual(g[0], 0.5, 1e-10));
		ASSERT("g[1] = 0.01/0.01 = 1.0", approxEqual(g[1], 1.0, 1e-10));
		ASSERT("g[2] = -0.005/0.02 = -0.25", approxEqual(g[2], -0.25, 1e-10));

		// Growth rate
		double G = glades::Kelly::factorGrowthRate(g, alphas, vars);
		// G = sum_j alpha_j^2 / (2*var_j)
		double expected = 0.02 * 0.02 / (2 * 0.04) + 0.01 * 0.01 / (2 * 0.01) +
		                  0.005 * 0.005 / (2 * 0.02);
		ASSERT("factor growth rate", approxEqual(G, expected, 1e-10));
	}

	// ---- Test 4: Growth attribution ----
	printf("\n--- Test 4: Growth attribution ---\n");
	{
		std::vector<double> alphas;
		alphas.push_back(0.02);
		alphas.push_back(0.01);

		std::vector<double> vars;
		vars.push_back(0.04);
		vars.push_back(0.01);

		std::vector<double> attr = glades::Kelly::factorGrowthAttribution(alphas, vars);
		ASSERT("attribution count", attr.size() == 2);
		ASSERT("attr[0] = 0.005", approxEqual(attr[0], 0.005, 1e-10));
		ASSERT("attr[1] = 0.005", approxEqual(attr[1], 0.005, 1e-10));

		// Sum should equal total optimal growth rate
		double totalG = 0.0;
		for (size_t i = 0; i < attr.size(); ++i)
			totalG += attr[i];
		std::vector<double> g = glades::Kelly::factorFractions(alphas, vars);
		double G = glades::Kelly::factorGrowthRate(g, alphas, vars);
		ASSERT("attribution sums to total growth", approxEqual(totalG, G, 1e-10));
	}

	// ---- Test 5: Cholesky solve ----
	printf("\n--- Test 5: Cholesky solve ---\n");
	{
		// 2x2 SPD matrix: [[4, 2], [2, 3]]
		std::vector<double> A;
		A.push_back(4.0); A.push_back(2.0);
		A.push_back(2.0); A.push_back(3.0);

		std::vector<double> b;
		b.push_back(1.0);
		b.push_back(2.0);

		std::vector<double> x = glades::Kelly::choleskySolve(A, b, 2);
		ASSERT("Cholesky solve found solution", !x.empty());

		// Verify: A*x == b
		std::vector<double> Ax = glades::Kelly::matVecMul(A, x, 2);
		ASSERT("Ax[0] ~ b[0]", approxEqual(Ax[0], b[0], 1e-10));
		ASSERT("Ax[1] ~ b[1]", approxEqual(Ax[1], b[1], 1e-10));
	}

	// ---- Test 6: Unconstrained multivariate Kelly ----
	printf("\n--- Test 6: Unconstrained multivariate Kelly ---\n");
	{
		size_t N = 3;
		// Diagonal covariance for simplicity
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;

		std::vector<double> mu;
		mu.push_back(0.02);
		mu.push_back(0.01);
		mu.push_back(-0.005);

		std::vector<double> w = glades::Kelly::unconstrainedWeights(mu, cov, N);
		ASSERT("unconstrained weights found", !w.empty());
		// w_i = mu_i / sigma_i^2 for diagonal case
		ASSERT("w[0] = 0.5", approxEqual(w[0], 0.5, 1e-8));
		ASSERT("w[1] = 1.0", approxEqual(w[1], 1.0, 1e-8));
		ASSERT("w[2] = -0.25", approxEqual(w[2], -0.25, 1e-8));
	}

	// ---- Test 7: Portfolio growth rate ----
	printf("\n--- Test 7: Portfolio growth rate ---\n");
	{
		size_t N = 2;
		std::vector<double> cov(4, 0.0);
		cov[0] = 0.04; cov[3] = 0.01;

		std::vector<double> mu;
		mu.push_back(0.02);
		mu.push_back(0.01);

		std::vector<double> w = glades::Kelly::unconstrainedWeights(mu, cov, N);
		double G = glades::Kelly::portfolioGrowthRate(w, mu, cov, N);

		// For diagonal case, G = sum_i mu_i^2 / (2*sigma_i^2)
		double expected = 0.02 * 0.02 / (2 * 0.04) + 0.01 * 0.01 / (2 * 0.01);
		ASSERT("portfolio growth rate", approxEqual(G, expected, 1e-10));
	}

	// ---- Test 8: Fractional Kelly ----
	printf("\n--- Test 8: Fractional Kelly ---\n");
	{
		std::vector<double> fullW;
		fullW.push_back(0.5);
		fullW.push_back(1.0);
		fullW.push_back(-0.25);

		std::vector<double> halfW = glades::Kelly::fractionalKelly(fullW, 0.5);
		ASSERT("half-Kelly[0]", approxEqual(halfW[0], 0.25, 1e-10));
		ASSERT("half-Kelly[1]", approxEqual(halfW[1], 0.5, 1e-10));
		ASSERT("half-Kelly[2]", approxEqual(halfW[2], -0.125, 1e-10));
	}

	// ---- Test 9: Cholesky with non-diagonal matrix ----
	printf("\n--- Test 9: Non-diagonal Cholesky ---\n");
	{
		size_t N = 3;
		// A positive definite matrix
		std::vector<double> A(9);
		A[0] = 4; A[1] = 2; A[2] = 1;
		A[3] = 2; A[4] = 5; A[5] = 3;
		A[6] = 1; A[7] = 3; A[8] = 6;

		std::vector<double> b;
		b.push_back(1.0);
		b.push_back(2.0);
		b.push_back(3.0);

		std::vector<double> x = glades::Kelly::choleskySolve(A, b, N);
		ASSERT("non-diagonal Cholesky succeeded", !x.empty());

		std::vector<double> Ax = glades::Kelly::matVecMul(A, x, N);
		for (size_t i = 0; i < N; ++i)
			ASSERT("Ax == b", approxEqual(Ax[i], b[i], 1e-8));
	}

	printf("\n  Kelly Criterion: All tests passed.\n\n");
}

// ============================================================================
// QP SOLVER UNIT TESTS
// ============================================================================

void QPSolverUnitTest(bool saveImages)
{
	printf("============================================================\n");
	printf("  QP Solver Unit Tests\n");
	printf("============================================================\n");

	// ---- Test 1: Budget-only constraint (closed form) ----
	printf("\n--- Test 1: Budget-only constraint ---\n");
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;

		std::vector<double> mu;
		mu.push_back(0.02);
		mu.push_back(0.01);
		mu.push_back(0.005);

		glades::QPResult result = glades::QPSolver::solveBudgetOnly(mu, cov, N);
		ASSERT("converged", result.converged);
		ASSERT("weights found", result.weights.size() == N);

		// Check budget constraint: sum = 1
		double sum = 0.0;
		for (size_t i = 0; i < N; ++i)
			sum += result.weights[i];
		ASSERT("budget constraint satisfied", approxEqual(sum, 1.0, 1e-8));

		// Check that growth rate is positive
		ASSERT("positive growth rate", result.objectiveValue > 0.0);

		printf("  w = [%.4f, %.4f, %.4f], G = %.6f, lambda = %.6f\n",
		       result.weights[0], result.weights[1], result.weights[2],
		       result.objectiveValue, result.budgetMultiplier);
	}

	// ---- Test 2: Budget + box constraints ----
	printf("\n--- Test 2: Budget + box constraints ---\n");
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;

		std::vector<double> mu;
		mu.push_back(0.02);
		mu.push_back(0.01);
		mu.push_back(0.005);

		glades::QPConstraints constraints;
		constraints.budgetConstraint = true;
		constraints.budgetTarget = 1.0;
		constraints.boxConstraints = true;
		constraints.lowerBounds.resize(N, 0.0);  // Long-only
		constraints.upperBounds.resize(N, 0.5);   // Max 50% per asset

		glades::QPResult result = glades::QPSolver::solve(mu, cov, N, constraints);
		ASSERT("converged", result.converged);

		// Check budget constraint
		double sum = 0.0;
		for (size_t i = 0; i < N; ++i)
			sum += result.weights[i];
		ASSERT("budget constraint", approxEqual(sum, 1.0, 1e-4));

		// Check box constraints
		for (size_t i = 0; i < N; ++i)
		{
			ASSERT("lower bound", result.weights[i] >= -1e-6);
			ASSERT("upper bound", result.weights[i] <= 0.5 + 1e-6);
		}

		printf("  w = [%.4f, %.4f, %.4f], G = %.6f\n",
		       result.weights[0], result.weights[1], result.weights[2],
		       result.objectiveValue);
	}

	// ---- Test 3: Equal expected returns -> equal weights ----
	printf("\n--- Test 3: Equal returns -> equal weights ---\n");
	{
		size_t N = 4;
		std::vector<double> cov(N * N, 0.0);
		for (size_t i = 0; i < N; ++i)
			cov[i * N + i] = 0.01;

		std::vector<double> mu(N, 0.01);

		glades::QPResult result = glades::QPSolver::solveBudgetOnly(mu, cov, N);
		ASSERT("converged", result.converged);

		// With identical assets, weights should be equal: 0.25 each
		for (size_t i = 0; i < N; ++i)
			ASSERT("equal weight", approxEqual(result.weights[i], 0.25, 1e-6));
	}

	// ---- Test 4: Constrained growth <= unconstrained growth ----
	printf("\n--- Test 4: Constrained <= unconstrained growth ---\n");
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;
		// Add some correlation
		cov[1] = 0.005; cov[3] = 0.005;
		cov[2] = 0.002; cov[6] = 0.002;
		cov[5] = 0.003; cov[7] = 0.003;

		std::vector<double> mu;
		mu.push_back(0.03);
		mu.push_back(0.02);
		mu.push_back(0.01);

		// Unconstrained Kelly
		std::vector<double> wUnc = glades::Kelly::unconstrainedWeights(mu, cov, N);
		double gUnc = glades::Kelly::portfolioGrowthRate(wUnc, mu, cov, N);

		// Budget-only
		glades::QPResult rBudget = glades::QPSolver::solveBudgetOnly(mu, cov, N);

		// Budget + box
		glades::QPConstraints constraints;
		constraints.budgetConstraint = true;
		constraints.boxConstraints = true;
		constraints.lowerBounds.resize(N, 0.0);
		constraints.upperBounds.resize(N, 0.6);
		glades::QPResult rBox = glades::QPSolver::solve(mu, cov, N, constraints);

		ASSERT("unconstrained >= budget-only", gUnc >= rBudget.objectiveValue - 1e-8);
		ASSERT("budget-only >= box-constrained",
		       rBudget.objectiveValue >= rBox.objectiveValue - 1e-6);

		printf("  G_unc=%.6f, G_budget=%.6f, G_box=%.6f\n",
		       gUnc, rBudget.objectiveValue, rBox.objectiveValue);
	}

	// ---- Image: Constrained vs unconstrained allocation comparison ----
	if (saveImages)
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;
		cov[1] = 0.005; cov[3] = 0.005;
		cov[2] = 0.002; cov[6] = 0.002;
		cov[5] = 0.003; cov[7] = 0.003;

		std::vector<double> mu;
		mu.push_back(0.03);
		mu.push_back(0.02);
		mu.push_back(0.01);

		std::vector<double> wUnc = glades::Kelly::unconstrainedWeights(mu, cov, N);
		glades::QPResult rBudget = glades::QPSolver::solveBudgetOnly(mu, cov, N);

		glades::QPConstraints cstr;
		cstr.budgetConstraint = true;
		cstr.boxConstraints = true;
		cstr.lowerBounds.resize(N, 0.0);
		cstr.upperBounds.resize(N, 0.6);
		glades::QPResult rBox = glades::QPSolver::solve(mu, cov, N, cstr);

		// Bar-chart style: 3 groups of 3 bars (use scatter with offset x positions)
		std::vector<std::string> names;
		names.push_back("Unconstrained Kelly");
		names.push_back("Budget-Only Kelly");
		names.push_back("Box-Constrained Kelly");

		std::vector<std::vector<double> > series(3);
		// Pack as [asset0_unc, asset0_bud, asset0_box, asset1_unc, ...]
		// Simpler: 3 series of length 3 (one per method)
		for (size_t i = 0; i < N; ++i)
		{
			series[0].push_back(wUnc[i]);
			series[1].push_back(rBudget.weights[i]);
			series[2].push_back(rBox.weights[i]);
		}

		std::vector<shmea::RGBA> colors;
		colors.push_back(shmea::RGBA(0xFF, 0x44, 0x44, 0xFF));
		colors.push_back(shmea::RGBA(0x44, 0x88, 0xFF, 0xFF));
		colors.push_back(shmea::RGBA(0x44, 0xCC, 0x44, 0xFF));

		saveMultiLineChart("qp_weight_comparison.png",
		                   "QP Solver: Portfolio Weights Comparison",
		                   "Asset Index", "Weight",
		                   names, series, colors);
	}

	// ---- Test 5: Zero expected returns -> minimum variance ----
	printf("\n--- Test 5: Zero returns -> min variance ---\n");
	{
		size_t N = 2;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[3] = 0.01;

		std::vector<double> mu(N, 0.0);

		glades::QPResult result = glades::QPSolver::solveBudgetOnly(mu, cov, N);
		ASSERT("converged", result.converged);

		// With zero returns and budget = 1, Kelly minimizes variance
		// w1*sigma1^2 = w2*sigma2^2 for diagonal case
		// w1*0.04 = w2*0.01, w1+w2=1 => w1 = 0.2, w2 = 0.8
		ASSERT("min variance w1", approxEqual(result.weights[0], 0.2, 1e-4));
		ASSERT("min variance w2", approxEqual(result.weights[1], 0.8, 1e-4));
	}

	// ---- Test 6: Leverage constraint ----
	printf("\n--- Test 6: Leverage constraint ---\n");
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;

		std::vector<double> mu;
		mu.push_back(0.05);
		mu.push_back(0.03);
		mu.push_back(-0.01);

		glades::QPConstraints constraints;
		constraints.budgetConstraint = true;
		constraints.leverageConstraint = true;
		constraints.maxLeverage = 1.0; // No shorting effectively

		glades::QPResult result = glades::QPSolver::solve(mu, cov, N, constraints);
		ASSERT("converged", result.converged);

		double l1norm = 0.0;
		for (size_t i = 0; i < N; ++i)
			l1norm += std::fabs(result.weights[i]);
		ASSERT("leverage constraint", l1norm <= 1.0 + 1e-4);

		printf("  w = [%.4f, %.4f, %.4f], |w|_1 = %.4f\n",
		       result.weights[0], result.weights[1], result.weights[2], l1norm);
	}

	// ---- Test 7: Single asset -> weight = 1 ----
	printf("\n--- Test 7: Single asset ---\n");
	{
		size_t N = 1;
		std::vector<double> cov(1, 0.04);
		std::vector<double> mu(1, 0.01);

		glades::QPResult result = glades::QPSolver::solveBudgetOnly(mu, cov, N);
		ASSERT("converged", result.converged);
		ASSERT("single asset weight = 1", approxEqual(result.weights[0], 1.0, 1e-8));
	}

	// ---- Test 8: Verify KKT multiplier sign ----
	printf("\n--- Test 8: KKT multiplier diagnostics ---\n");
	{
		size_t N = 3;
		std::vector<double> cov(N * N, 0.0);
		cov[0] = 0.04; cov[4] = 0.01; cov[8] = 0.02;

		std::vector<double> mu;
		mu.push_back(0.02);
		mu.push_back(0.01);
		mu.push_back(0.005);

		glades::QPConstraints constraints;
		constraints.budgetConstraint = true;
		constraints.boxConstraints = true;
		constraints.lowerBounds.resize(N, 0.05); // Min 5% each
		constraints.upperBounds.resize(N, 0.8);

		glades::QPResult result = glades::QPSolver::solve(mu, cov, N, constraints);
		ASSERT("converged", result.converged);
		ASSERT("box multipliers sized", result.boxMultipliersLower.size() == N);

		printf("  Budget multiplier (lambda): %.6f\n", result.budgetMultiplier);
		printf("  Growth rate: %.6f\n", result.objectiveValue);
	}

	printf("\n  QP Solver: All tests passed.\n\n");
}
