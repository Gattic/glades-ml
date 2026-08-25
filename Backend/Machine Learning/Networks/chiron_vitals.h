// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// VITALS: read-only CHIRON training telemetry.  The implementation is kept
// independent of the trainer so every estimator and self-test can be exercised
// deterministically on CPU.  CUDA taps only populate the samples defined here.
#ifndef GLADES_CHIRON_VITALS_H
#define GLADES_CHIRON_VITALS_H

#include <stddef.h>
#include <string>
#include <vector>

namespace glades {
namespace chiron {
namespace vitals {

static const int kQuantileCount = 6;
static const int kFrequencyBands = 8;
static const int kPositionBands = 4;
static const int kCopyBands = 2;
static const int kStrata = kFrequencyBands * kPositionBands * kCopyBands;
static const int kAdamStatCount = 16;

enum AdamStatSlot
{
	ADAM_COUNT = 0,
	ADAM_M_SATURATED = 1,
	ADAM_V_SATURATED = 2,
	ADAM_DEAD_UPDATE = 3,
	ADAM_SIGN_AGREE = 4,
	ADAM_SIGNAL_POWER = 5,
	ADAM_NOISE_POWER = 6,
	ADAM_EFFICIENCY_SUM = 7,
	ADAM_NOISE_FIT = 8,
	ADAM_UPDATE_SQ = 9,
	ADAM_WEIGHT_SQ = 10,
	ADAM_REL_UPDATE_SUM = 11,
	ADAM_REL_UPDATE_MAX = 12,
	ADAM_RESERVED_13 = 13,
	ADAM_RESERVED_14 = 14,
	ADAM_RESERVED_15 = 15
};

struct Config
{
	bool enabled;
	int cadence;
	int probeEvery;
	int warmupSteps;
	int robustWindow;
	float piedDropout;
	float observerWarn;
	float observerCritical;
	float ledgerTolerance;
	float mixtureTolerance;
	float slopeTolerance;

	Config();
};

struct LayerEnergyTap
{
	// Means over all sampled (micro-step, token, channel) elements.
	double pBefore;
	double pAfter;
	double qEnergy;
	double incrementEnergy;
	double committedEnergy;
	double alignment;
	LayerEnergyTap();
};

struct AdamTap
{
	// Additive device reductions using AdamStatSlot.  Multiple parameter
	// groups may be summed into one tap or retained as separate taps.
	double values[kAdamStatCount];
	std::string group;
	AdamTap();
};

struct ProbeSample
{
	bool valid;
	double canaryLoss;
	double freshLoss;
	double baselineGap;
	double oldCanaryLoss;
	double oldFreshLoss;
	double oldBaselineGap;
	double samBaseLoss;
	double samPerturbedLoss;
	double atomicsVarianceFloor;
	double replaySpacing;
	std::vector<float> deltaDt;
	std::vector<float> delta2Dt;
	std::vector<float> maskLosses;
	std::vector<float> repeatLogpCopy;
	std::vector<float> repeatLogpTruth;
	ProbeSample();
};

struct StepSample
{
	int step;
	unsigned long long tokensSeen;
	int sequenceLength;
	int vocabularySize;
	int accumSteps;
	double loss;
	double previousLoss;
	double learningRate;
	double clipScale;
	double microOneGradNormSq;
	double fullGradNormSq;
	std::vector<double> microOneGroupNormSq;
	std::vector<double> fullGroupNormSq;
	double maskVariancePrediction;

	// V1 boundaries are indexed 0..L; all layer vectors are indexed 0..L-1.
	std::vector<double> dqMeanSq;
	std::vector<double> dpMeanSq;
	std::vector<double> clampFraction;
	std::vector<double> nonfiniteFraction;
	std::vector<float> observerResidual;
	std::vector<LayerEnergyTap> energy;
	std::vector<double> fisherByLayer;
	std::vector<AdamTap> optimizer;

	// Output-measure taps. nll/logZ/top1/tokens all have sequenceLength items.
	std::vector<int> tokens;
	std::vector<float> nll;
	std::vector<float> logZ;
	std::vector<unsigned char> top1;
	std::vector<unsigned long long> frozenUnigram;
	std::vector<double> coverageMass;

	ProbeSample probe;
	StepSample();
};

struct Stratum
{
	int frequencyBand;
	int positionBand;
	int copyAvailable;
	unsigned long count;
	double weight;
	double mean;
	double quantiles[kQuantileCount];
	double slope;
	double slopeSE;
	bool saturated;
	double remainingExtractable;
	Stratum();
};

struct Alarm
{
	std::string term;
	std::string severity;
	std::string code;
	double value;
	double threshold;
	Alarm();
	Alarm(const char* t, const char* s, const char* c, double v, double th);
};

struct Report
{
	int step;
	// V1
	std::vector<double> adjointGain;
	double depthLogGain;
	std::vector<double> dpFlatness;
	std::vector<double> clampBind;
	// V2
	double observerQ999;
	double observerMax;
	unsigned long observerWarnCount;
	unsigned long observerCriticalCount;
	// V3
	double ledgerClosureRelative;
	double cancellationFraction;
	double maskCalibration;
	double totalIncrementEnergy;
	double energyLogSlope;
	// V4
	double clipScale;
	double signalClippedFraction;
	// V5
	double adamMSaturationFraction;
	double adamVSaturationFraction;
	double adamDeadFraction;
	double adamSignAgreement;
	double adamRelativeUpdateRms;
	double adamRelativeUpdateMax;
	// V6
	double noisePower;
	double signalPower;
	double noiseScale;
	std::vector<double> groupNoisePower;
	std::vector<double> groupSignalPower;
	std::vector<double> groupNoiseScale;
	// V7
	double optimizerEfficiency;
	double noiseFitRate;
	double memorizationLedger;
	// V8/V9
	double piedSusceptibility;
	double piedMcVariance;
	double piedMcRatio;
	double samSharpness;
	// V10-V12
	std::vector<Stratum> strata;
	double mixtureReconstructionError;
	double tokenUtilityPerBillion;
	double slopeReconstructionRelative;
	bool stopConsider;
	double copyGap;
	double headTailGap;
	double positionGapNoCopy;
	// V13/V14
	double memorizationGap;
	double oldMemorizationGap;
	double transportDrift;
	double transportDiffusion;
	double peclet;
	// V15
	double coverageParticipation;
	std::vector<double> coverageBandShare;
	// V16
	double hillTailIndex;
	double batchZ;
	double logZMean;
	double logZQ99;
	// V17/V18
	double copyGainShare;
	double repeatPreference;
	double contextualGain;
	// Continuous plumbing tests.
	bool ledgerClosurePass;
	bool maskCalibrationPass;
	bool dpFlatnessPass;
	bool mixtureReconstructionPass;
	bool slopeReconstructionPass;
	bool valid;
	std::vector<Alarm> alarms;

	Report();
};

// Stateful host reducer.  It stores only bounded scalar/stratum history; raw
// token and activation vectors are never retained after observe().
class Tracker
{
public:
	explicit Tracker(const Config& cfg = Config());
	~Tracker();
	void reset();
	Report observe(const StepSample& sample);
	const Config& config() const { return cfg_; }

private:
	struct History;
	Config cfg_;
	std::vector<History> history_;
	std::vector<double> lambdaHistory_;
	std::vector<double> energyStepHistory_;
	std::vector<double> energyLogHistory_;
	std::vector<double> hillHistory_;
	std::vector<double> microNormHistory_;
	std::vector<double> fullNormHistory_;
	std::vector< std::vector<double> > groupMicroHistory_;
	std::vector< std::vector<double> > groupFullHistory_;
	double memorizationLedger_;
	int lambdaAlarmRun_;
	int stopConsiderRun_;
};

// Public deterministic estimator helpers used by trainer smoke tests and unit
// tests.  Empty/invalid input returns NaN rather than silently fabricating 0.
double quantile(std::vector<double> values, double p);
double median(std::vector<double> values);
double mad(const std::vector<double>& values, double center);
double sample_variance(const std::vector<float>& values);
double hill_tail_index(const std::vector<float>& losses, size_t k);
void gradient_noise_pair(double microOneNormSq, double fullNormSq,
                         double& signalPower, double& noisePower);
void drift_diffusion(const std::vector<float>& deltaDt,
                     const std::vector<float>& delta2Dt,
                     double spacing, double atomicsVarianceFloor,
                     double& drift, double& diffusion, double& peclet);
std::vector<unsigned char> copy_labels(const std::vector<int>& tokens, int ngram);

} // namespace vitals
} // namespace chiron
} // namespace glades

#endif
