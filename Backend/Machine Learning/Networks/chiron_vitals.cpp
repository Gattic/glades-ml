// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
#include "chiron_vitals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace glades {
namespace chiron {
namespace vitals {

namespace {

static double qnan()
{
	return std::numeric_limits<double>::quiet_NaN();
}

static bool finite_d(double x)
{
	return x == x && x != std::numeric_limits<double>::infinity()
	       && x != -std::numeric_limits<double>::infinity();
}

static double safe_ratio(double a, double b)
{
	return finite_d(a) && finite_d(b) && std::fabs(b) > 1e-30 ? a / b : qnan();
}

static int position_band(int t, int T)
{
	if (T <= 1) return 0;
	if (t < T / 16) return 0;
	if (t < T / 4) return 1;
	if (t < T / 2) return 2;
	return 3;
}

static int frequency_band(unsigned long long c, unsigned long long maxCount)
{
	if (c == 0 || maxCount == 0) return kFrequencyBands - 1;
	const double octaves = std::log((double)maxCount / (double)c) / std::log(2.0);
	int b = (int)std::floor(octaves);
	if (b < 0) b = 0;
	if (b >= kFrequencyBands) b = kFrequencyBands - 1;
	return b;
}

static int stratum_id(int fb, int pb, int cp)
{
	return (fb * kPositionBands + pb) * kCopyBands + cp;
}

static double mean_float(const std::vector<float>& v)
{
	if (v.empty()) return qnan();
	double s = 0.0;
	for (size_t i = 0; i < v.size(); ++i) s += (double)v[i];
	return s / (double)v.size();
}

static void ols(const std::vector<double>& x, const std::vector<double>& y,
                double& slope, double& slopeSE)
{
	slope = qnan(); slopeSE = qnan();
	if (x.size() != y.size() || x.size() < 3) return;
	double mx = 0.0, my = 0.0;
	for (size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
	mx /= (double)x.size(); my /= (double)y.size();
	double sxx = 0.0, sxy = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
	{
		const double dx = x[i] - mx;
		sxx += dx * dx;
		sxy += dx * (y[i] - my);
	}
	if (sxx <= 0.0) return;
	slope = sxy / sxx;
	const double intercept = my - slope * mx;
	double rss = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
	{
		const double r = y[i] - intercept - slope * x[i];
		rss += r * r;
	}
	slopeSE = std::sqrt((rss / (double)(x.size() - 2)) / sxx);
}

// Fit y=A+B*z over a small family of exponential and power-law bases.  Model
// selection uses the final 30% prediction error; only the asymptotic gap is
// returned, so the inherently weak extrapolation remains explicitly advisory.
static double fit_remaining(const std::vector<double>& x,
                            const std::vector<double>& y)
{
	if (x.size() != y.size() || x.size() < 6) return qnan();
	const size_t cut = (size_t)std::floor(0.7 * (double)x.size());
	if (cut < 3 || cut >= x.size()) return qnan();
	const double x0 = x.front();
	const double span = std::max(1.0, x.back() - x0);
	double best = std::numeric_limits<double>::infinity();
	double bestA = qnan();
	for (int family = 0; family < 2; ++family)
	{
		for (int gi = 0; gi < 24; ++gi)
		{
			const double shape = family == 0
			    ? std::pow(10.0, -1.0 + 2.0 * (double)gi / 23.0)
			    : 0.05 + 0.95 * (double)gi / 23.0;
			double mz = 0.0, my = 0.0;
			std::vector<double> z(cut, 0.0);
			for (size_t i = 0; i < cut; ++i)
			{
				const double xn = (x[i] - x0) / span;
				z[i] = family == 0 ? std::exp(-xn / shape)
				                    : std::pow(1.0 + xn, -shape);
				mz += z[i]; my += y[i];
			}
			mz /= (double)cut; my /= (double)cut;
			double zz = 0.0, zy = 0.0;
			for (size_t i = 0; i < cut; ++i)
			{
				zz += (z[i] - mz) * (z[i] - mz);
				zy += (z[i] - mz) * (y[i] - my);
			}
			if (zz <= 1e-30) continue;
			const double B = zy / zz;
			const double A = my - B * mz;
			double err = 0.0;
			for (size_t i = cut; i < x.size(); ++i)
			{
				const double xn = (x[i] - x0) / span;
				const double zi = family == 0 ? std::exp(-xn / shape)
				                              : std::pow(1.0 + xn, -shape);
				const double r = y[i] - (A + B * zi);
				err += r * r;
			}
			if (err < best) { best = err; bestA = A; }
		}
	}
	return finite_d(bestA) ? std::max(0.0, y.back() - bestA) : qnan();
}

static double max_abs_deviation_one(const std::vector<double>& v)
{
	double out = 0.0;
	for (size_t i = 0; i < v.size(); ++i)
		if (finite_d(v[i])) out = std::max(out, std::fabs(v[i] - 1.0));
	return out;
}

static void aggregate_adam(const std::vector<AdamTap>& taps, double out[kAdamStatCount])
{
	for (int j = 0; j < kAdamStatCount; ++j) out[j] = 0.0;
	for (size_t i = 0; i < taps.size(); ++i)
		for (int j = 0; j < kAdamStatCount; ++j)
			out[j] += taps[i].values[j];
}

static double logz_quantile(const std::vector<float>& v, double p)
{
	std::vector<double> d;
	d.reserve(v.size());
	for (size_t i = 0; i < v.size(); ++i) if (finite_d(v[i])) d.push_back(v[i]);
	return quantile(d, p);
}

} // anonymous namespace

Config::Config()
    : enabled(false), cadence(25), probeEvery(100), warmupSteps(1500),
      robustWindow(500), piedDropout(0.1f), observerWarn(0.1f),
      observerCritical(0.7f), ledgerTolerance(1e-3f),
      mixtureTolerance(0.005f), slopeTolerance(0.10f) {}

LayerEnergyTap::LayerEnergyTap()
    : pBefore(qnan()), pAfter(qnan()), qEnergy(qnan()),
      incrementEnergy(0.0), committedEnergy(0.0), alignment(0.0) {}

AdamTap::AdamTap() : group()
{
	for (int i = 0; i < kAdamStatCount; ++i) values[i] = 0.0;
}

ProbeSample::ProbeSample()
    : valid(false), canaryLoss(qnan()), freshLoss(qnan()), baselineGap(0.0),
      oldCanaryLoss(qnan()), oldFreshLoss(qnan()), oldBaselineGap(0.0),
      samBaseLoss(qnan()), samPerturbedLoss(qnan()), atomicsVarianceFloor(0.0),
      replaySpacing(1.0) {}

StepSample::StepSample()
    : step(0), tokensSeen(0), sequenceLength(0), vocabularySize(0), accumSteps(1),
      loss(qnan()), previousLoss(qnan()), learningRate(0.0), clipScale(1.0),
      microOneGradNormSq(qnan()), fullGradNormSq(qnan()),
      maskVariancePrediction(qnan()) {}

Stratum::Stratum()
    : frequencyBand(0), positionBand(0), copyAvailable(0), count(0), weight(0.0),
      mean(qnan()), slope(qnan()), slopeSE(qnan()), saturated(false),
      remainingExtractable(qnan())
{
	for (int i = 0; i < kQuantileCount; ++i) quantiles[i] = qnan();
}

Alarm::Alarm() : term(), severity(), code(), value(qnan()), threshold(qnan()) {}
Alarm::Alarm(const char* t, const char* s, const char* c, double v, double th)
    : term(t), severity(s), code(c), value(v), threshold(th) {}

Report::Report()
    : step(0), depthLogGain(qnan()), observerQ999(qnan()), observerMax(qnan()),
      observerWarnCount(0), observerCriticalCount(0), ledgerClosureRelative(qnan()),
      cancellationFraction(qnan()), maskCalibration(qnan()), totalIncrementEnergy(qnan()),
      energyLogSlope(qnan()), clipScale(1.0), signalClippedFraction(qnan()), adamMSaturationFraction(qnan()),
      adamVSaturationFraction(qnan()), adamDeadFraction(qnan()), adamSignAgreement(qnan()),
      adamRelativeUpdateRms(qnan()), adamRelativeUpdateMax(qnan()), noisePower(qnan()),
      signalPower(qnan()), noiseScale(qnan()), optimizerEfficiency(qnan()),
      noiseFitRate(qnan()), memorizationLedger(0.0), piedSusceptibility(qnan()),
      piedMcVariance(qnan()), piedMcRatio(qnan()), samSharpness(qnan()),
      mixtureReconstructionError(qnan()), tokenUtilityPerBillion(qnan()),
      slopeReconstructionRelative(qnan()), stopConsider(false), copyGap(qnan()), headTailGap(qnan()),
      positionGapNoCopy(qnan()), memorizationGap(qnan()), oldMemorizationGap(qnan()),
      transportDrift(qnan()), transportDiffusion(qnan()), peclet(qnan()),
      coverageParticipation(qnan()), hillTailIndex(qnan()), batchZ(qnan()),
      logZMean(qnan()), logZQ99(qnan()), copyGainShare(qnan()),
      repeatPreference(qnan()), contextualGain(qnan()), ledgerClosurePass(false),
      maskCalibrationPass(false), dpFlatnessPass(false),
      mixtureReconstructionPass(false), slopeReconstructionPass(false), valid(false) {}

double quantile(std::vector<double> values, double p)
{
	std::vector<double> clean;
	clean.reserve(values.size());
	for (size_t i = 0; i < values.size(); ++i)
		if (finite_d(values[i])) clean.push_back(values[i]);
	values.swap(clean);
	if (values.empty()) return qnan();
	std::sort(values.begin(), values.end());
	if (p <= 0.0) return values.front();
	if (p >= 1.0) return values.back();
	const double pos = p * (double)(values.size() - 1);
	const size_t lo = (size_t)std::floor(pos);
	const size_t hi = (size_t)std::ceil(pos);
	const double a = pos - (double)lo;
	return values[lo] * (1.0 - a) + values[hi] * a;
}

double median(std::vector<double> values) { return quantile(values, 0.5); }

double mad(const std::vector<double>& values, double center)
{
	std::vector<double> d;
	d.reserve(values.size());
	for (size_t i = 0; i < values.size(); ++i)
		if (finite_d(values[i])) d.push_back(std::fabs(values[i] - center));
	return median(d);
}

double sample_variance(const std::vector<float>& values)
{
	if (values.size() < 2) return qnan();
	double m = 0.0;
	for (size_t i = 0; i < values.size(); ++i) m += values[i];
	m /= (double)values.size();
	double s = 0.0;
	for (size_t i = 0; i < values.size(); ++i)
	{
		const double d = (double)values[i] - m;
		s += d * d;
	}
	return s / (double)(values.size() - 1);
}

double hill_tail_index(const std::vector<float>& losses, size_t k)
{
	std::vector<double> x;
	x.reserve(losses.size());
	for (size_t i = 0; i < losses.size(); ++i)
		if (finite_d(losses[i]) && losses[i] > 0.0f) x.push_back(losses[i]);
	if (x.size() < 3) return qnan();
	std::sort(x.begin(), x.end(), std::greater<double>());
	if (k < 2) k = 2;
	if (k >= x.size()) k = x.size() - 1;
	const double threshold = x[k];
	if (threshold <= 0.0) return qnan();
	double denom = 0.0;
	for (size_t i = 0; i < k; ++i) denom += std::log(x[i] / threshold);
	return denom > 0.0 ? (double)k / denom : qnan();
}

void gradient_noise_pair(double microOneNormSq, double fullNormSq,
                         double& signalPower, double& noisePower)
{
	if (!finite_d(microOneNormSq) || !finite_d(fullNormSq))
	{
		signalPower = noisePower = qnan(); return;
	}
	noisePower = (4.0 / 3.0) * (microOneNormSq - fullNormSq);
	signalPower = (4.0 * fullNormSq - microOneNormSq) / 3.0;
}

void drift_diffusion(const std::vector<float>& deltaDt,
                     const std::vector<float>& delta2Dt,
                     double spacing, double atomicsVarianceFloor,
                     double& drift, double& diffusion, double& peclet)
{
	drift = diffusion = peclet = qnan();
	if (deltaDt.empty() || delta2Dt.empty() || spacing <= 0.0) return;
	const double mean1 = mean_float(deltaDt);
	const double v1 = sample_variance(deltaDt);
	const double v2 = sample_variance(delta2Dt);
	if (!finite_d(v1) || !finite_d(v2)) return;
	const double a = std::max(0.0, (v2 - 2.0 * v1) / (2.0 * spacing * spacing));
	const double b = std::max(0.0, (v1 - a * spacing * spacing) / spacing);
	drift = mean1 / spacing;
	diffusion = b;
	const double corrected = std::max(0.0, v1 - atomicsVarianceFloor);
	peclet = corrected > 0.0 ? std::fabs(mean1) / std::sqrt(corrected) : qnan();
}

std::vector<unsigned char> copy_labels(const std::vector<int>& tokens, int ngram)
{
	std::vector<unsigned char> out(tokens.size(), 0u);
	if (ngram < 2 || tokens.size() < (size_t)ngram) return out;
	std::map<unsigned long long, int> seen;
	for (size_t end = (size_t)ngram - 1; end < tokens.size(); ++end)
	{
		unsigned long long h = 1469598103934665603ULL;
		for (int j = ngram - 1; j >= 0; --j)
		{
			h ^= (unsigned long long)(unsigned int)tokens[end - (size_t)j] + 0x9e3779b9ULL;
			h *= 1099511628211ULL;
		}
		if (seen.find(h) != seen.end()) out[end] = 1u;
		seen[h] = (int)end;
	}
	return out;
}

struct Tracker::History
{
	double tokens;
	double loss;
	std::vector<double> mean;
	std::vector<double> weight;
	History() : tokens(0.0), loss(qnan()), mean(kStrata, qnan()), weight(kStrata, 0.0) {}
};

Tracker::Tracker(const Config& cfg)
    : cfg_(cfg), memorizationLedger_(0.0), lambdaAlarmRun_(0), stopConsiderRun_(0) {}

Tracker::~Tracker() {}

void Tracker::reset()
{
	history_.clear(); lambdaHistory_.clear(); energyStepHistory_.clear(); energyLogHistory_.clear();
	hillHistory_.clear(); microNormHistory_.clear(); fullNormHistory_.clear();
	groupMicroHistory_.clear(); groupFullHistory_.clear();
	memorizationLedger_ = 0.0; lambdaAlarmRun_ = 0; stopConsiderRun_ = 0;
}

Report Tracker::observe(const StepSample& sample)
{
	Report r;
	r.step = sample.step;
	r.clipScale = sample.clipScale;
	const size_t boundaries = std::min(sample.dqMeanSq.size(), sample.dpMeanSq.size());

	// V1 — depth gain and dp profile.
	if (boundaries >= 2)
	{
		std::vector<double> g(boundaries, qnan()), p(boundaries, qnan());
		for (size_t i = 0; i < boundaries; ++i)
		{
			g[i] = sample.dqMeanSq[i] >= 0.0 ? std::sqrt(sample.dqMeanSq[i]) : qnan();
			p[i] = sample.dpMeanSq[i] >= 0.0 ? std::sqrt(sample.dpMeanSq[i]) : qnan();
		}
		for (size_t l = 1; l < boundaries; ++l)
			r.adjointGain.push_back(safe_ratio(g[l - 1], g[l]));
		r.depthLogGain = (g.front() > 0.0 && g.back() > 0.0)
		                   ? std::log(g.front() / g.back()) : qnan();
		for (size_t l = 0; l < boundaries; ++l)
			r.dpFlatness.push_back(safe_ratio(p[l], p.back()));
		r.clampBind = sample.clampFraction;
		r.dpFlatnessPass = max_abs_deviation_one(r.dpFlatness) <= 0.25;
	}

	// V2 — observer residual tails.
	if (!sample.observerResidual.empty())
	{
		std::vector<double> residual;
		residual.reserve(sample.observerResidual.size());
		for (size_t i = 0; i < sample.observerResidual.size(); ++i)
		{
			const double v = sample.observerResidual[i];
			if (!finite_d(v)) continue;
			residual.push_back(v);
			if (v >= cfg_.observerWarn) ++r.observerWarnCount;
			if (v >= cfg_.observerCritical) ++r.observerCriticalCount;
		}
		r.observerQ999 = quantile(residual, 0.999);
		r.observerMax = quantile(residual, 1.0);
	}

	// V3 — energy closure, cancellation and PIED mask calibration.
	if (!sample.energy.empty())
	{
		double e = 0.0, ec = 0.0, c = 0.0;
		for (size_t l = 0; l < sample.energy.size(); ++l)
		{
			e += sample.energy[l].incrementEnergy;
			ec += sample.energy[l].committedEnergy;
			c += sample.energy[l].alignment;
		}
		r.totalIncrementEnergy = e;
		r.cancellationFraction = ec > 0.0 ? -2.0 * c / ec : qnan();
		r.maskCalibration = e > 0.0 ? ec / e : qnan();
		// Check each shear locally. This is algebraically equivalent to the
		// catalog's telescoping identity when no intervening WhiSC rotation
		// changes p, and remains exact for the production shear->rotate order.
		double closureAbs = 0.0, closureScale = 0.0;
		for (size_t l = 0; l < sample.energy.size(); ++l)
		{
			const LayerEnergyTap& z = sample.energy[l];
			if (!finite_d(z.pBefore) || !finite_d(z.pAfter)) continue;
			closureAbs += std::fabs(z.pAfter - z.pBefore
			                        - z.committedEnergy - 2.0 * z.alignment);
			closureScale += std::fabs(z.pAfter);
		}
		r.ledgerClosureRelative = closureScale > 1e-30 ? closureAbs / closureScale : qnan();
		r.ledgerClosurePass = finite_d(r.ledgerClosureRelative)
		                       && r.ledgerClosureRelative < cfg_.ledgerTolerance;
		const double expectedMask = cfg_.piedDropout > 0.0f
		    ? 1.0 / (1.0 - (double)cfg_.piedDropout) : 1.0;
		r.maskCalibrationPass = finite_d(r.maskCalibration)
		                        && std::fabs(r.maskCalibration - expectedMask)
		                           <= std::max(0.02, 0.05 * expectedMask);
		if (e > 0.0)
		{
			energyStepHistory_.push_back((double)sample.step);
			energyLogHistory_.push_back(std::log(e));
			if (energyLogHistory_.size() > 100u)
			{
				energyLogHistory_.erase(energyLogHistory_.begin());
				energyStepHistory_.erase(energyStepHistory_.begin());
			}
			double se=qnan(); ols(energyStepHistory_, energyLogHistory_, r.energyLogSlope, se);
		}
	}

	// V6 and V4's signal-clipped fraction. Report the ratio of window means;
	// the single-step pair is intentionally not interpreted (it is often
	// negative under anisotropic noise, as the design warns).
	if (finite_d(sample.microOneGradNormSq) && finite_d(sample.fullGradNormSq))
	{
		microNormHistory_.push_back(sample.microOneGradNormSq);
		fullNormHistory_.push_back(sample.fullGradNormSq);
		if (microNormHistory_.size() > 100u) microNormHistory_.erase(microNormHistory_.begin());
		if (fullNormHistory_.size() > 100u) fullNormHistory_.erase(fullNormHistory_.begin());
	}
	double microMean = 0.0, fullMean = 0.0;
	for (size_t i=0;i<microNormHistory_.size();++i) microMean += microNormHistory_[i];
	for (size_t i=0;i<fullNormHistory_.size();++i) fullMean += fullNormHistory_[i];
	if (!microNormHistory_.empty()) microMean /= (double)microNormHistory_.size(); else microMean=qnan();
	if (!fullNormHistory_.empty()) fullMean /= (double)fullNormHistory_.size(); else fullMean=qnan();
	gradient_noise_pair(microMean, fullMean, r.signalPower, r.noisePower);
	r.noiseScale = r.signalPower > 0.0 ? r.noisePower / r.signalPower : qnan();
	const size_t nGradGroups = std::min(sample.microOneGroupNormSq.size(),
	                                    sample.fullGroupNormSq.size());
	if (groupMicroHistory_.size() < nGradGroups)
	{
		groupMicroHistory_.resize(nGradGroups); groupFullHistory_.resize(nGradGroups);
	}
	for (size_t g = 0; g < nGradGroups; ++g)
	{
		if (finite_d(sample.microOneGroupNormSq[g]) && finite_d(sample.fullGroupNormSq[g]))
		{
			groupMicroHistory_[g].push_back(sample.microOneGroupNormSq[g]);
			groupFullHistory_[g].push_back(sample.fullGroupNormSq[g]);
			if (groupMicroHistory_[g].size()>100u) groupMicroHistory_[g].erase(groupMicroHistory_[g].begin());
			if (groupFullHistory_[g].size()>100u) groupFullHistory_[g].erase(groupFullHistory_[g].begin());
		}
		double mm=0.0,fm=0.0;
		for(size_t i=0;i<groupMicroHistory_[g].size();++i) mm+=groupMicroHistory_[g][i];
		for(size_t i=0;i<groupFullHistory_[g].size();++i) fm+=groupFullHistory_[g][i];
		if(!groupMicroHistory_[g].empty())mm/=(double)groupMicroHistory_[g].size();else mm=qnan();
		if(!groupFullHistory_[g].empty())fm/=(double)groupFullHistory_[g].size();else fm=qnan();
		double gs = qnan(), gn = qnan(); gradient_noise_pair(mm, fm, gs, gn);
		r.groupSignalPower.push_back(gs); r.groupNoisePower.push_back(gn);
		r.groupNoiseScale.push_back(gs > 0.0 ? gn / gs : qnan());
	}
	const double denomV4 = r.signalPower + r.noisePower / 4.0;
	r.signalClippedFraction = denomV4 > 0.0
	    ? (1.0 - sample.clipScale) * r.signalPower / denomV4 : qnan();

	// V5/V7 — fused Adam reductions.
	double adam[kAdamStatCount]; aggregate_adam(sample.optimizer, adam);
	if (adam[ADAM_COUNT] > 0.0)
	{
		const double n = adam[ADAM_COUNT];
		r.adamMSaturationFraction = adam[ADAM_M_SATURATED] / n;
		r.adamVSaturationFraction = adam[ADAM_V_SATURATED] / n;
		r.adamDeadFraction = adam[ADAM_DEAD_UPDATE] / n;
		r.adamSignAgreement = adam[ADAM_SIGN_AGREE] / n;
		r.adamRelativeUpdateRms = std::sqrt(std::max(0.0, adam[ADAM_UPDATE_SQ])
		                                    / std::max(1e-30, adam[ADAM_WEIGHT_SQ]));
		r.adamRelativeUpdateMax = adam[ADAM_REL_UPDATE_MAX];
		r.optimizerEfficiency = adam[ADAM_EFFICIENCY_SUM] / n;
		r.noiseFitRate = adam[ADAM_NOISE_FIT];
		if (finite_d(r.noiseFitRate)) memorizationLedger_ += r.noiseFitRate;
	}
	r.memorizationLedger = memorizationLedger_;

	// V8/V9.
	if (!sample.fisherByLayer.empty())
	{
		double f = 0.0;
		for (size_t i = 0; i < sample.fisherByLayer.size(); ++i) f += sample.fisherByLayer[i];
		const double varEta = cfg_.piedDropout > 0.0f
		    ? (double)cfg_.piedDropout / (1.0 - (double)cfg_.piedDropout) : 0.0;
		r.piedSusceptibility = varEta * f;
	}
	if (!sample.probe.maskLosses.empty())
	{
		r.piedMcVariance = sample_variance(sample.probe.maskLosses);
		r.piedMcRatio = safe_ratio(r.piedMcVariance, r.piedSusceptibility);
	}
	if (finite_d(sample.probe.samBaseLoss) && finite_d(sample.probe.samPerturbedLoss))
		r.samSharpness = sample.probe.samPerturbedLoss - sample.probe.samBaseLoss;

	// V10 — deterministic token strata and exact mixture reconstruction.
	std::vector< std::vector<double> > sv(kStrata);
	std::vector<unsigned char> copy8 = copy_labels(sample.tokens, 8);
	std::vector<unsigned char> copy2 = copy_labels(sample.tokens, 2);
	unsigned long long maxFreq = 0;
	for (size_t i = 0; i < sample.frozenUnigram.size(); ++i)
		maxFreq = std::max(maxFreq, sample.frozenUnigram[i]);
	const size_t ntok = std::min(sample.tokens.size(), sample.nll.size());
	for (size_t t = 0; t < ntok; ++t)
	{
		const int tok = sample.tokens[t];
		const unsigned long long fc = tok >= 0 && (size_t)tok < sample.frozenUnigram.size()
		    ? sample.frozenUnigram[(size_t)tok] : 0ULL;
		const int fb = frequency_band(fc, maxFreq);
		const int pb = position_band((int)t, sample.sequenceLength > 0
		                                     ? sample.sequenceLength : (int)ntok);
		const int cp = ((t < copy8.size() && copy8[t]) || (t < copy2.size() && copy2[t])) ? 1 : 0;
		if (finite_d(sample.nll[t])) sv[(size_t)stratum_id(fb, pb, cp)].push_back(sample.nll[t]);
	}
	static const double qp[kQuantileCount] = {0.10, 0.25, 0.50, 0.75, 0.90, 0.99};
	double mix = 0.0;
	for (int sid = 0; sid < kStrata; ++sid)
	{
		Stratum st;
		st.copyAvailable = sid % kCopyBands;
		st.positionBand = (sid / kCopyBands) % kPositionBands;
		st.frequencyBand = sid / (kCopyBands * kPositionBands);
		st.count = (unsigned long)sv[(size_t)sid].size();
		st.weight = ntok > 0 ? (double)st.count / (double)ntok : 0.0;
		if (!sv[(size_t)sid].empty())
		{
			double sum = 0.0;
			for (size_t j = 0; j < sv[(size_t)sid].size(); ++j) sum += sv[(size_t)sid][j];
			st.mean = sum / (double)sv[(size_t)sid].size();
			for (int qi = 0; qi < kQuantileCount; ++qi)
				st.quantiles[qi] = quantile(sv[(size_t)sid], qp[qi]);
			mix += st.weight * st.mean;
		}
		r.strata.push_back(st);
	}
	const double tokenMean = mean_float(sample.nll);
	r.mixtureReconstructionError = finite_d(tokenMean) ? std::fabs(mix - tokenMean) : qnan();
	r.mixtureReconstructionPass = finite_d(r.mixtureReconstructionError)
	                                  && r.mixtureReconstructionError < cfg_.mixtureTolerance;

	// V12 raw gaps.
	std::vector<double> cp0, cp1, head, tail, earlyNoCopy, lateNoCopy;
	for (int sid = 0; sid < kStrata; ++sid)
	{
		const Stratum& st = r.strata[(size_t)sid];
		std::vector<double>& values = sv[(size_t)sid];
		if (st.copyAvailable) cp1.insert(cp1.end(), values.begin(), values.end());
		else cp0.insert(cp0.end(), values.begin(), values.end());
		if (st.frequencyBand == 0) head.insert(head.end(), values.begin(), values.end());
		if (st.frequencyBand == kFrequencyBands - 1) tail.insert(tail.end(), values.begin(), values.end());
		if (!st.copyAvailable && st.positionBand == 0) earlyNoCopy.insert(earlyNoCopy.end(), values.begin(), values.end());
		if (!st.copyAvailable && st.positionBand == kPositionBands - 1) lateNoCopy.insert(lateNoCopy.end(), values.begin(), values.end());
	}
	r.copyGap = median(cp0) - median(cp1);
	r.headTailGap = median(tail) - median(head);
	r.positionGapNoCopy = median(earlyNoCopy) - median(lateNoCopy);

	// Append bounded scalar/stratum history, then estimate V11 slopes.
	History hp;
	hp.tokens = (double)sample.tokensSeen;
	hp.loss = finite_d(tokenMean) ? tokenMean : sample.loss;
	for (int sid = 0; sid < kStrata; ++sid)
	{
		hp.mean[(size_t)sid] = r.strata[(size_t)sid].mean;
		hp.weight[(size_t)sid] = r.strata[(size_t)sid].weight;
	}
	history_.push_back(hp);
	const size_t keep = (size_t)std::max(8, cfg_.robustWindow);
	if (history_.size() > keep) history_.erase(history_.begin(), history_.begin() + (history_.size() - keep));
	std::vector<double> hx, hy;
	for (size_t i = 0; i < history_.size(); ++i)
		if (finite_d(history_[i].loss)) { hx.push_back(history_[i].tokens); hy.push_back(history_[i].loss); }
	double globalSlope = qnan(), globalSE = qnan();
	ols(hx, hy, globalSlope, globalSE);
	r.tokenUtilityPerBillion = finite_d(globalSlope) ? -globalSlope * 1e9 : qnan();
	for (int sid = 0; sid < kStrata; ++sid)
	{
		std::vector<double> x, y;
		for (size_t i = 0; i < history_.size(); ++i)
			if (finite_d(history_[i].mean[(size_t)sid]))
			{
				x.push_back(history_[i].tokens);
				y.push_back(history_[i].mean[(size_t)sid]);
			}
		ols(x, y, r.strata[(size_t)sid].slope, r.strata[(size_t)sid].slopeSE);
		r.strata[(size_t)sid].saturated = finite_d(r.strata[(size_t)sid].slope)
		    && finite_d(r.strata[(size_t)sid].slopeSE)
		    && std::fabs(r.strata[(size_t)sid].slope) <= 1.96 * r.strata[(size_t)sid].slopeSE;
		r.strata[(size_t)sid].remainingExtractable = fit_remaining(x, y);
	}
	if (history_.size() >= 2)
	{
		const History& a = history_.front();
		const History& b = history_.back();
		const double dt = b.tokens - a.tokens;
		if (dt > 0.0)
		{
			double reconDelta = 0.0;
			for (int sid = 0; sid < kStrata; ++sid)
			{
				const double ma = a.mean[(size_t)sid], mb = b.mean[(size_t)sid];
				if (!finite_d(ma) || !finite_d(mb)) continue;
				const double wbar = 0.5 * (a.weight[(size_t)sid] + b.weight[(size_t)sid]);
				const double mbar = 0.5 * (ma + mb);
				reconDelta += wbar * (mb - ma) + (b.weight[(size_t)sid] - a.weight[(size_t)sid]) * mbar;
			}
			const double actual = b.loss - a.loss;
			r.slopeReconstructionRelative = std::fabs(actual) > 1e-12
			    ? std::fabs(reconDelta - actual) / std::fabs(actual)
			    : std::fabs(reconDelta - actual);
			r.slopeReconstructionPass = r.slopeReconstructionRelative < cfg_.slopeTolerance;
		}
	}

	// V13/V14 probes.
	if (sample.probe.valid)
	{
		r.memorizationGap = (sample.probe.freshLoss - sample.probe.canaryLoss)
		                     - sample.probe.baselineGap;
		r.oldMemorizationGap = (sample.probe.oldFreshLoss - sample.probe.oldCanaryLoss)
		                        - sample.probe.oldBaselineGap;
	}
	drift_diffusion(sample.probe.deltaDt, sample.probe.delta2Dt,
	                sample.probe.replaySpacing, sample.probe.atomicsVarianceFloor,
	                r.transportDrift, r.transportDiffusion, r.peclet);
	// V11 pre-registered 20B stop consideration: evaluate only on 5k-step
	// boundaries and require three consecutive windows.
	if (sample.step > 0 && (sample.step % 5000) == 0)
	{
		bool remaining = false;
		for (size_t i=0;i<r.strata.size();++i)
			if (finite_d(r.strata[i].remainingExtractable) && r.strata[i].remainingExtractable > 0.02)
				remaining = true;
		const bool candidate = finite_d(r.tokenUtilityPerBillion)
		    && r.tokenUtilityPerBillion < 0.005 && finite_d(r.peclet)
		    && r.peclet < 0.5 && !remaining;
		stopConsiderRun_ = candidate ? stopConsiderRun_ + 1 : 0;
	}
	r.stopConsider = stopConsiderRun_ >= 3;

	// V15 coverage.
	if (!sample.coverageMass.empty())
	{
		double sum = 0.0, sq = 0.0;
		for (size_t i = 0; i < sample.coverageMass.size(); ++i)
		{
			sum += sample.coverageMass[i]; sq += sample.coverageMass[i] * sample.coverageMass[i];
		}
		r.coverageParticipation = sq > 0.0
		    ? sum * sum / ((double)sample.coverageMass.size() * sq) : 0.0;
		r.coverageBandShare.assign(kFrequencyBands, 0.0);
		for (size_t i = 0; i < sample.coverageMass.size(); ++i)
		{
			const unsigned long long fc = i < sample.frozenUnigram.size() ? sample.frozenUnigram[i] : 0ULL;
			r.coverageBandShare[(size_t)frequency_band(fc, maxFreq)] += sample.coverageMass[i];
		}
		if (sum > 0.0) for (int b = 0; b < kFrequencyBands; ++b) r.coverageBandShare[(size_t)b] /= sum;
	}

	// V16 tail, batch z-score and normalizer.
	r.hillTailIndex = hill_tail_index(sample.nll, std::min((size_t)512, ntok > 1 ? ntok - 1 : (size_t)0));
	if (!sample.logZ.empty())
	{
		r.logZMean = mean_float(sample.logZ);
		r.logZQ99 = logz_quantile(sample.logZ, 0.99);
	}
	if (history_.size() > 5 && finite_d(hp.loss))
	{
		std::vector<double> prev;
		for (size_t i = 0; i + 1 < history_.size(); ++i) prev.push_back(history_[i].loss);
		const double med = median(prev), scale = 1.4826 * mad(prev, med);
		r.batchZ = scale > 1e-12 ? (hp.loss - med) / scale : qnan();
	}

	// V17 copy-gain share from the previous observed composition-fixed pair.
	if (history_.size() >= 2)
	{
		const History& a = history_[history_.size() - 2];
		const History& b = history_.back();
		double cg = 0.0, ng = 0.0;
		for (int sid = 0; sid < kStrata; ++sid)
		{
			if (!finite_d(a.mean[(size_t)sid]) || !finite_d(b.mean[(size_t)sid])) continue;
			const double gain = b.weight[(size_t)sid] * (a.mean[(size_t)sid] - b.mean[(size_t)sid]);
			if (sid % kCopyBands) cg += gain; else ng += gain;
		}
		r.copyGainShare = safe_ratio(cg, cg + ng);
	}
	if (sample.probe.repeatLogpCopy.size() == sample.probe.repeatLogpTruth.size()
	    && !sample.probe.repeatLogpCopy.empty())
	{
		double sum = 0.0;
		for (size_t i = 0; i < sample.probe.repeatLogpCopy.size(); ++i)
			sum += (double)sample.probe.repeatLogpCopy[i] - sample.probe.repeatLogpTruth[i];
		r.repeatPreference = sum / (double)sample.probe.repeatLogpCopy.size();
	}

	// V18 contextual gain relative to a frozen add-one unigram model.
	if (ntok > 0 && !sample.frozenUnigram.empty())
	{
		unsigned long long total = 0ULL;
		for (size_t i = 0; i < sample.frozenUnigram.size(); ++i) total += sample.frozenUnigram[i];
		double gain = 0.0; size_t used = 0;
		const double den = (double)total + (double)sample.frozenUnigram.size();
		for (size_t t = 0; t < ntok; ++t)
		{
			const int tok = sample.tokens[t];
			if (tok < 0 || (size_t)tok >= sample.frozenUnigram.size()) continue;
			const double su = -std::log(((double)sample.frozenUnigram[(size_t)tok] + 1.0) / den);
			gain += su - sample.nll[t]; ++used;
		}
		r.contextualGain = used ? gain / (double)used : qnan();
	}

	// Typed alarms. FAST uses a trailing robust band; SLOW/self-test alarms use
	// frozen absolute design thresholds so secular drift cannot normalize away.
	if (finite_d(r.depthLogGain))
	{
		if (lambdaHistory_.size() >= 20 && sample.step >= cfg_.warmupSteps)
		{
			const double med = median(lambdaHistory_);
			const double scale = 1.4826 * mad(lambdaHistory_, med);
			const double th = med + 6.0 * std::max(scale, 1e-6);
			const bool forcingStationary = r.maskCalibrationPass
			    && (!finite_d(r.energyLogSlope) || std::fabs(r.energyLogSlope) <= 5e-5);
			if (r.depthLogGain > th && forcingStationary) ++lambdaAlarmRun_;
			else lambdaAlarmRun_ = 0;
			if (lambdaAlarmRun_ >= 3)
				r.alarms.push_back(Alarm("I", "FAST", "V1_DEPTH_GAIN", r.depthLogGain, th));
		}
		lambdaHistory_.push_back(r.depthLogGain);
		if (lambdaHistory_.size() > (size_t)cfg_.robustWindow) lambdaHistory_.erase(lambdaHistory_.begin());
	}
	if (finite_d(r.observerMax) && r.observerMax >= cfg_.observerCritical)
		r.alarms.push_back(Alarm("II", "FAST", "V2_OBSERVER_CRITICAL", r.observerMax, cfg_.observerCritical));
	else if (finite_d(r.observerQ999) && r.observerQ999 >= cfg_.observerWarn)
		r.alarms.push_back(Alarm("II", "SLOW", "V2_OBSERVER_WARN", r.observerQ999, cfg_.observerWarn));
	if (!r.ledgerClosurePass && finite_d(r.ledgerClosureRelative))
		r.alarms.push_back(Alarm("SELF", "CRITICAL", "V3_LEDGER_CLOSURE", r.ledgerClosureRelative, cfg_.ledgerTolerance));
	if (!r.maskCalibrationPass && finite_d(r.maskCalibration))
		r.alarms.push_back(Alarm("SELF", "CRITICAL", "V3_MASK_CALIBRATION", r.maskCalibration,
		                         cfg_.piedDropout > 0.0 ? 1.0 / (1.0 - cfg_.piedDropout) : 1.0));
	if (!r.dpFlatnessPass && !r.dpFlatness.empty())
		r.alarms.push_back(Alarm("SELF", "WARN", "V1_DP_FLATNESS", max_abs_deviation_one(r.dpFlatness), 0.25));
	if (!r.mixtureReconstructionPass && finite_d(r.mixtureReconstructionError))
		r.alarms.push_back(Alarm("SELF", "CRITICAL", "V10_MIXTURE_RECONSTRUCTION",
		                         r.mixtureReconstructionError, cfg_.mixtureTolerance));
	if (finite_d(r.energyLogSlope) && energyLogHistory_.size() >= 80u && r.energyLogSlope > 5e-5)
		r.alarms.push_back(Alarm("I", "SLOW", "V3_ENERGY_INFLATION", r.energyLogSlope, 5e-5));
	if (finite_d(r.adamMSaturationFraction) && (r.adamMSaturationFraction > 0.01 || r.adamVSaturationFraction > 0.01))
		r.alarms.push_back(Alarm("III", "SLOW", "V5_ADAM_SATURATION",
		                         std::max(r.adamMSaturationFraction, r.adamVSaturationFraction), 0.01));
	if (finite_d(r.peclet) && r.peclet < 0.5 && sample.step >= cfg_.warmupSteps)
		r.alarms.push_back(Alarm("IV", "SLOW", "V14_PECLET_COLLAPSE", r.peclet, 0.5));
	if (finite_d(r.batchZ) && std::fabs(r.batchZ) > 6.0)
		r.alarms.push_back(Alarm("IV", "FAST", "V16_HARD_BATCH", std::fabs(r.batchZ), 6.0));
	if (finite_d(r.hillTailIndex))
	{
		if (hillHistory_.size() >= 20u)
		{
			const double q5 = quantile(hillHistory_, 0.05);
			if (r.hillTailIndex < q5)
				r.alarms.push_back(Alarm("IV", "FAST", "V16_HEAVY_TAIL", r.hillTailIndex, q5));
		}
		hillHistory_.push_back(r.hillTailIndex);
		if (hillHistory_.size() > (size_t)cfg_.robustWindow) hillHistory_.erase(hillHistory_.begin());
	}

	r.valid = boundaries >= 2 || ntok > 0 || !sample.optimizer.empty();
	return r;
}

} // namespace vitals
} // namespace chiron
} // namespace glades
