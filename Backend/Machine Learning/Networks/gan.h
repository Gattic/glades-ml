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
#ifndef _GLADES_GAN
#define _GLADES_GAN

#include "network.h"
#include "training_config.h"
#include "../nnetwork_status.h"
#include "../rng.h"
#include <vector>
#include <string>
#include <stdint.h>

namespace glades {

class DataInput;

struct GANConfig
{
	enum LossType { GAN_VANILLA = 0, GAN_WGAN_GP = 1, GAN_LSGAN = 2 };
	enum ArchType { GAN_DFF = 0, GAN_CNN = 1 };
	enum VariantType { GAN_STANDARD = 0, GAN_INFO = 1, GAN_STYLE = 2, GAN_CYCLE = 3 };

	LossType lossType;
	ArchType archType;
	VariantType variantType;

	// Training ratios
	int nCriticPerGenerator; // discriminator steps per generator step (1 for vanilla, 5 for WGAN-GP)
	int nGenPerCritic;       // generator steps per discriminator step (default: 1)
	float labelSmoothing;    // one-sided label smoothing for real targets (default: 0, typical: 0.1)

	// WGAN-GP specific
	float gpLambda; // gradient penalty coefficient (default: 10)

	// Optimizer (Adam, applied to both networks)
	float generatorLR;
	float discriminatorLR;
	float adamBeta1;
	float adamBeta2;
	float adamEps;

	// Noise
	unsigned int noiseDim; // latent space dimension (z)

	// Generator output activation
	bool generatorOutputSigmoid; // apply sigmoid to generator output (for [0,1] data like images)
	bool generatorLayerNorm;     // apply layer normalization to generator hidden layers

	// CNN-specific (generator output / discriminator input dimensions)
	CNNConfig generatorCNN;
	CNNConfig discriminatorCNN;
	DeconvConfig generatorDeconv;

	// Training
	int epochs;
	int batchSize;

	// InfoGAN-specific
	struct InfoConfig
	{
		unsigned int numCategorical; // one-hot categorical code dims (default 10)
		unsigned int numContinuous;  // continuous code dims (default 2)
		float infoLambda;            // MI loss weight (default 1.0)
		float catScale;              // categorical code input scale (default 1.0)
		float divLambda;             // mode-seeking diversity loss weight (default 0.0 = off)

		InfoConfig() : numCategorical(10u), numContinuous(2u), infoLambda(1.0f), catScale(1.0f), divLambda(0.0f) {}
	};
	InfoConfig infoConfig;

	// StyleGAN-specific
	struct StyleConfig
	{
		float noiseScaleInit;      // per-layer noise scale init (default 0.0)
		unsigned int mappingLayers; // mapping network depth (default 4)
		unsigned int mappingWidth;  // mapping network width (default 128)

		StyleConfig() : noiseScaleInit(0.0f), mappingLayers(4u), mappingWidth(128u) {}
	};
	StyleConfig styleConfig;

	// CycleGAN-specific
	struct CycleConfig
	{
		float cycleLambda;    // cycle consistency loss weight (default 10.0)
		float identityLambda; // identity loss weight (default 5.0)

		CycleConfig() : cycleLambda(10.0f), identityLambda(5.0f) {}
	};
	CycleConfig cycleConfig;

	// Composable feature flags (can be combined freely)
	bool useInfo;   // enable InfoGAN features
	bool useStyle;  // enable StyleGAN features
	bool useCycle;  // enable CycleGAN features

	// Adaptive disc/gen balancing
	bool adaptiveBalance;          // skip disc updates when disc loss < threshold
	float adaptiveDiscThreshold;   // disc loss threshold (default: 0.1 for LSGAN, ~0.5 for vanilla)

	// Spectral normalization
	bool spectralNorm;             // apply spectral norm to discriminator weights
	int spectralNormIters;         // power iteration count (default: 1)

	// Gradient clipping
	float gradClipNorm;            // max L2 norm for generator gradients (0 = off, default: 0)

	// Weight decay (L2 regularization)
	float weightDecay;             // L2 weight decay coefficient (0 = off, default: 0)

	// Generator output temperature
	float genOutputTemp;           // sigmoid temperature for deconv output (default: 1.0, higher = softer)

	// Parallelism
	bool deterministicReduce; // ordered gradient reduction for reproducibility (default: true)

	GANConfig()
		: lossType(GAN_VANILLA),
		  archType(GAN_DFF),
		  variantType(GAN_STANDARD),
		  nCriticPerGenerator(1),
		  nGenPerCritic(1),
		  labelSmoothing(0.0f),
		  gpLambda(10.0f),
		  generatorLR(0.0002f),
		  discriminatorLR(0.0002f),
		  adamBeta1(0.5f),
		  adamBeta2(0.999f),
		  adamEps(1e-8f),
		  noiseDim(100u),
		  generatorOutputSigmoid(false),
		  generatorLayerNorm(false),
		  generatorCNN(),
		  discriminatorCNN(),
		  generatorDeconv(),
		  epochs(100),
		  batchSize(32),
		  infoConfig(),
		  styleConfig(),
		  cycleConfig(),
		  useInfo(false),
		  useStyle(false),
		  useCycle(false),
		  adaptiveBalance(false),
		  adaptiveDiscThreshold(0.1f),
		  spectralNorm(false),
		  spectralNormIters(1),
		  gradClipNorm(0.0f),
		  weightDecay(0.0f),
		  genOutputTemp(1.0f),
		  deterministicReduce(true)
	{
	}
};

struct GANEpochMetrics
{
	int epoch;
	float dLossReal;    // discriminator loss on real data
	float dLossFake;    // discriminator loss on fake data
	float gLoss;        // generator loss
	float wasserstein;  // Wasserstein estimate (WGAN only)

	// InfoGAN-specific
	float infoLoss;     // mutual information loss
	float catAccuracy;  // categorical code classification accuracy
	float divLoss;      // mode-seeking diversity loss

	// CycleGAN-specific
	float cycleLoss;    // cycle consistency loss
	float identityLoss; // identity loss
	float dLossA;       // discriminator A loss
	float dLossB;       // discriminator B loss
	float gLossAB;      // generator A->B loss
	float gLossBA;      // generator B->A loss

	// Health diagnostics
	float dOutReal;        // mean discriminator output on real data
	float dOutFake;        // mean discriminator output on fake data
	float genGradNorm;     // L2 norm of generator gradient
	float sampleDiversity; // mean per-pixel variance across generated samples (0 = mode collapse)

	GANEpochMetrics()
		: epoch(0), dLossReal(0.0f), dLossFake(0.0f), gLoss(0.0f), wasserstein(0.0f),
		  infoLoss(0.0f), catAccuracy(0.0f), divLoss(0.0f),
		  cycleLoss(0.0f), identityLoss(0.0f), dLossA(0.0f), dLossB(0.0f),
		  gLossAB(0.0f), gLossBA(0.0f),
		  dOutReal(0.0f), dOutFake(0.0f), genGradNorm(0.0f), sampleDiversity(0.0f)
	{
	}
};

class IGANCallbacks
{
public:
	virtual ~IGANCallbacks() {}
	virtual void onEpochEnd(const GANEpochMetrics&) {}
	virtual bool shouldStop(const GANEpochMetrics&) { return false; }
};

struct GradientBuffer
{
	// DFF gradient arrays: [transition][weights/biases]
	std::vector<std::vector<float> > dffGW;
	std::vector<std::vector<float> > dffGBias;

	// CNN conv gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > convGW;
	std::vector<std::vector<float> > convGBias;
	// CNN FC gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > fcGW;
	std::vector<std::vector<float> > fcGBias;

	// Deconv FC gradient arrays
	std::vector<float> deconvFcGW;
	std::vector<float> deconvFcGBias;
	// Deconv layer gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > deconvGW;
	std::vector<std::vector<float> > deconvGBias;
	// Deconv batch norm gradient arrays: [layer][gamma/beta]
	std::vector<std::vector<float> > deconvGBnGamma;
	std::vector<std::vector<float> > deconvGBnBeta;

	// Q-head gradient arrays (InfoGAN)
	std::vector<float> qGW;
	std::vector<float> qGBias;

	// Generator-side Q-head gradient arrays (InfoGAN direct path)
	std::vector<float> genQGW;
	std::vector<float> genQGBias;
	std::vector<float> genQGHiddenW;
	std::vector<float> genQGHiddenBias;

	void initFromDFF(const NNetwork& net);
	void initFromCNN(const NNetwork& net);
	void initFromDeconv(const NNetwork& net);
	void initFromQHead(unsigned int sharedDim, unsigned int qOutDim);
	void initFromGenQHead(unsigned int sharedDim, unsigned int qOutDim, unsigned int hiddenDim);
	void zero();
	void addToDFF(NNetwork& net) const;
	void addToCNN(NNetwork& net) const;
	void addToDeconv(NNetwork& net) const;
};

struct DeconvScratchArena
{
	// Per-layer forward buffers
	std::vector<std::vector<float> > upsampled;   // [layer] inC*upH*upW
	std::vector<std::vector<float> > cols;         // [layer] N*K (im2col output)
	std::vector<std::vector<float> > outputCols;   // [layer] outK*N (transposed conv path)

	// Per-layer backward buffers
	std::vector<std::vector<float> > dCols;        // [layer] N*K
	std::vector<std::vector<float> > dUp;          // [layer] inC*upH*upW
	std::vector<std::vector<float> > dInput;       // [layer] inC*inH*inW

	// FC backward
	std::vector<float> dFC;

	// Backward dCur working buffer
	std::vector<float> dCur;

	bool initialized;
	DeconvScratchArena() : initialized(false) {}

	void initFromDeconv(const NNetwork& net);
};

struct GANThreadCtx
{
	GradientBuffer genGrads;
	GradientBuffer discGrads;
	GradientBuffer qGrads;
	GradientBuffer mappingGrads;

	// Per-thread style affine gradient arrays: [affineIdx][weights/biases]
	std::vector<std::vector<float> > styleGW, styleGBias;
	std::vector<float> gNoiseScalesLocal;

	// LayerNorm gradient arrays
	std::vector<std::vector<float> > lnGGamma, lnGBeta;

	// Activation scratch
	std::vector<std::vector<float> > genAct, discActReal, discActFake;
	std::vector<float> cnnOutReal, cnnOutFake;
	std::vector<std::vector<float> > cnnFcActFake; // cached FC activations for Q-head
	std::vector<float> deconvOut;
	std::vector<std::vector<float> > deconvScratch;
	// Pre-allocated deconv scratch arena (eliminates per-sample malloc)
	DeconvScratchArena deconvArena;

	// Pre-allocated InfoGAN per-sample scratch
	std::vector<float> penultActBuf;    // [penultDim] reusable
	std::vector<float> qInfoDFakeBuf;   // [dataDim] reusable
	std::vector<float> dFake, dWVec, genInput;

	// Style forward scratch
	std::vector<std::vector<float> > mapAct;
	std::vector<float> wVec;
	std::vector<std::vector<float> > sXNorms, sNoiseVecs;
	std::vector<float> sMeans, sInvStds;

	// InfoGAN scratch
	std::vector<float> qOut, qGrad, sharedGrad;
	std::vector<std::vector<float> > qDelta;
	std::vector<float> genQHiddenPre, genQHiddenPost, genQDHidden; // genQHead hidden layer scratch

	// dffBackwardStyled scratch (replaces class-member scratchDelta)
	std::vector<std::vector<float> > scratchDeltaLocal;

	// Discard buffer for generator phase disc backward
	GradientBuffer discardDiscGrads;

	// ---- CycleGAN-specific members ----
	GradientBuffer genBAGrads;
	GradientBuffer discBGrads;
	GradientBuffer qBGrads;
	GradientBuffer mappingBAGrads;
	GradientBuffer discardDiscBGrads;
	GradientBuffer discardGenGrads;   // discard buffer for cycle input-grad-only backward through G_AB

	// CycleGAN style BA gradient buffers
	std::vector<std::vector<float> > styleBAGW, styleBAGBias;
	std::vector<float> gNoiseScalesBALocal;

	// CycleGAN LN BA gradient buffers
	std::vector<std::vector<float> > lnBAGGamma, lnBAGBeta;

	// CycleGAN activation scratch (BA generator, second disc, cycle, identity)
	std::vector<std::vector<float> > genBAAct;
	std::vector<float> deconvOutBA;
	std::vector<std::vector<float> > deconvScratchBA;
	std::vector<float> genBAInput;
	std::vector<std::vector<float> > mapBAAct;
	std::vector<float> wBAVec;
	std::vector<std::vector<float> > sBAXNorms, sBANoiseVecs;
	std::vector<float> sBAMeans, sBAInvStds;
	std::vector<std::vector<float> > discBActReal, discBActFake;
	std::vector<float> cnnOutBReal, cnnOutBFake;

	// CycleGAN gen loop scratch: cycle reconstruction
	std::vector<std::vector<float> > recAAct, recBAct;
	std::vector<float> deconvOutRecA, deconvOutRecB;
	std::vector<std::vector<float> > deconvScratchRecA, deconvScratchRecB;
	// CycleGAN gen loop scratch: identity
	std::vector<std::vector<float> > identAAct, identBAct;
	std::vector<float> deconvOutIdentA, deconvOutIdentB;
	std::vector<std::vector<float> > deconvScratchIdentA, deconvScratchIdentB;
	// CycleGAN gen loop: disc forward on fakes
	std::vector<std::vector<float> > discBFakeAct, discAFakeAct;
	std::vector<float> cnnOutFakeBGen, cnnOutFakeAGen;
	// CycleGAN gen loop: extra dFake for genBA backward
	std::vector<float> dFakeA;

	// CycleGAN loss accumulators
	float gLossAB, gLossBA, cycleLoss, identityLoss;

	// Diversity loss scratch (mode-seeking: compare consecutive samples)
	std::vector<float> prevFake;

	// Loss accumulators
	float dLossReal, dLossFake, gLoss, wasserstein, infoLoss, divLoss;
	unsigned int catCorrect, catTotal;

	// Discriminator output accumulators (raw predictions before loss)
	float dOutRealSum, dOutFakeSum;

	GANThreadCtx()
		: gLossAB(0.0f), gLossBA(0.0f), cycleLoss(0.0f), identityLoss(0.0f),
		  dLossReal(0.0f), dLossFake(0.0f), gLoss(0.0f),
		  wasserstein(0.0f), infoLoss(0.0f), divLoss(0.0f),
		  catCorrect(0u), catTotal(0u),
		  dOutRealSum(0.0f), dOutFakeSum(0.0f) {}

	void zeroLosses();
	void zeroGrads();
};

class GAN
{
public:
	// Standard / InfoGAN constructor (backward compatible)
	GAN(const GANConfig& config,
	    const NNInfo* generatorInfo,
	    const NNInfo* discriminatorInfo);

	// StyleGAN constructor (needs mapping network info)
	GAN(const GANConfig& config,
	    const NNInfo* generatorInfo,
	    const NNInfo* discriminatorInfo,
	    const NNInfo* mappingInfo);

	// CycleGAN constructor (needs 4 network infos)
	GAN(const GANConfig& config,
	    const NNInfo* genABInfo,
	    const NNInfo* genBAInfo,
	    const NNInfo* discAInfo,
	    const NNInfo* discBInfo);

	// Cycle+Style constructor (needs 4 network infos + 2 mapping net infos)
	GAN(const GANConfig& config,
	    const NNInfo* genABInfo,
	    const NNInfo* genBAInfo,
	    const NNInfo* discAInfo,
	    const NNInfo* discBInfo,
	    const NNInfo* mapABInfo,
	    const NNInfo* mapBAInfo);

	~GAN();

	// Train the GAN on real data (standard, InfoGAN, StyleGAN)
	NNetworkStatus train(const DataInput* realData, IGANCallbacks* cb = NULL);

	// Train CycleGAN on two domains
	NNetworkStatus train(const DataInput* domainA, const DataInput* domainB, IGANCallbacks* cb = NULL);

	// Generate samples from noise
	// output: [numSamples][outputDim]
	NNetworkStatus generate(unsigned int numSamples,
	                        std::vector<std::vector<float> >& outSamples) const;

	// InfoGAN controlled generation (fixed latent codes)
	NNetworkStatus generateInfoGAN(unsigned int numSamples,
	                               const std::vector<float>* fixedCatCode,
	                               const std::vector<float>* fixedContCode,
	                               std::vector<std::vector<float> >& outSamples) const;

	// CycleGAN domain transfer
	NNetworkStatus translate(const DataInput* input, bool aToB,
	                         std::vector<std::vector<float> >& outSamples) const;

	// CycleGAN domain transfer with Info codes
	NNetworkStatus translate(const DataInput* input, bool aToB,
	                         const std::vector<float>* fixedCatCode,
	                         const std::vector<float>* fixedContCode,
	                         std::vector<std::vector<float> >& outSamples) const;

	// Access internal networks (for save/load)
	const NNetwork& getGenerator() const;
	const NNetwork& getDiscriminator() const;
	const NNetwork& getGeneratorBA() const;
	const NNetwork& getDiscriminatorB() const;

	// Save all network weights to binary files in the given directory.
	// Creates <dir>/<prefix>_genAB.bin, _genBA.bin, _discA.bin, _discB.bin
	// Returns the number of networks successfully saved.
	int saveWeights(const std::string& dir, const std::string& prefix) const;

	// Load all network weights from binary files in the given directory.
	// Reads <dir>/<prefix>_genAB.bin, _genBA.bin, _discA.bin, _discB.bin
	// Returns the number of networks successfully loaded.
	int loadWeights(const std::string& dir, const std::string& prefix);

	void setSeed(uint64_t seed);

private:
	// Non-copyable
	GAN(const GAN&);
	GAN& operator=(const GAN&);

	GANConfig config;
	NNetwork generator;
	NNetwork discriminator;
	glades::rng::Engine rngEngine;

	// Per-network Adam optimizer state
	struct AdamState
	{
		// DFF transitions: mW[t], vW[t], mBias[t], vBias[t]
		std::vector<std::vector<float> > mW;
		std::vector<std::vector<float> > vW;
		std::vector<std::vector<float> > mBias;
		std::vector<std::vector<float> > vBias;

		// CNN conv layers
		std::vector<std::vector<float> > mCW;
		std::vector<std::vector<float> > vCW;
		std::vector<std::vector<float> > mCBias;
		std::vector<std::vector<float> > vCBias;

		// CNN FC layers
		std::vector<std::vector<float> > mFW;
		std::vector<std::vector<float> > vFW;
		std::vector<std::vector<float> > mFBias;
		std::vector<std::vector<float> > vFBias;

		unsigned long long step;

		AdamState() : step(0ULL) {}
	};

	AdamState genAdamState;
	AdamState discAdamState;

	// Spectral normalization state (power iteration u vectors)
	struct SpectralNormState
	{
		// u vectors for power iteration, one per weight matrix
		// DFF: snU[transition][out_dim]
		// CNN conv: snConvU[layer][outC]
		// CNN FC: snFCU[layer][out_dim]
		std::vector<std::vector<float> > snU;
		std::vector<std::vector<float> > snConvU;
		std::vector<std::vector<float> > snFCU;
		bool initialized;
		SpectralNormState() : initialized(false) {}
	};

	SpectralNormState discSNState;
	SpectralNormState discBSNState; // CycleGAN

	// ---- InfoGAN members ----
	struct QNetworkHead
	{
		std::vector<float> W, bias, gW, gBias;
		unsigned int sharedDim, qOutDim;
		bool initialized;

		// Optional hidden layer (hiddenDim > 0 enables two-layer Q-head)
		unsigned int hiddenDim;
		std::vector<float> hiddenW, hiddenBias, gHiddenW, gHiddenBias;

		QNetworkHead() : sharedDim(0u), qOutDim(0u), initialized(false), hiddenDim(0u) {}
	};
	QNetworkHead qHead;
	AdamState qAdamState;

	// Direct generator-side Q-head (InfoGAN: bypasses discriminator for gen gradient)
	QNetworkHead genQHead;
	AdamState genQAdamState;

	// ---- StyleGAN members ----
	NNetwork mappingNet;
	AdamState mappingAdamState;

	struct StyleAffine
	{
		std::vector<float> W, bias, gW, gBias;
		unsigned int wDim, outDim;

		StyleAffine() : wDim(0u), outDim(0u) {}
	};
	std::vector<StyleAffine> styleAffines;
	AdamState styleAffineAdamState;
	std::vector<float> noiseScales, gNoiseScales;

	// ---- Layer Normalization for generator hidden layers ----
	struct LayerNormParams
	{
		// Per hidden-layer transition: gamma/beta affine params
		std::vector<std::vector<float> > gamma;  // [transition][out]
		std::vector<std::vector<float> > beta;
		std::vector<std::vector<float> > gGamma; // accumulated gradients
		std::vector<std::vector<float> > gBeta;

		// Adam state for gamma/beta
		std::vector<std::vector<float> > mGamma, vGamma;
		std::vector<std::vector<float> > mBeta, vBeta;
		unsigned long long step;

		// Forward cache (for backward pass)
		std::vector<std::vector<float> > zNorm;  // [transition][out] normalized values
		std::vector<float> invStd;               // [transition] inverse std dev

		bool initialized;

		LayerNormParams() : step(0ULL), initialized(false) {}
	};

	mutable LayerNormParams genLNParams;
	mutable LayerNormParams genBALNParams; // for CycleGAN

	// ---- CycleGAN members ----
	NNetwork generatorBA;
	NNetwork discriminatorB;
	AdamState genBAAdamState;
	AdamState discBAdamState;

	// ---- CycleGAN+StyleGAN members (BA style apparatus) ----
	NNetwork mappingNetBA;
	AdamState mappingBAAdamState;
	std::vector<StyleAffine> styleAffinesBA;
	AdamState styleAffineBAAdamState;
	std::vector<float> noiseScalesBA, gNoiseScalesBA;

	// ---- CycleGAN+InfoGAN members (discB Q-head) ----
	QNetworkHead qHeadB;
	AdamState qBAdamState;

	// ---- Training scratch buffers (reused across samples) ----
	std::vector<std::vector<float> > scratchDelta;

	// Initialize tensor parameters for both networks given data dimensions
	bool initDFFTensors(NNetwork& net, unsigned int inputSize, unsigned int outputSize);
	bool initCNNTensors(NNetwork& net, const CNNConfig& cnnCfg, unsigned int outputSize);
	void initAdamState(AdamState& state, const NNetwork& net);

	// DFF forward/backward
	void dffForward(const NNetwork& net, const float* input, unsigned int inputSize,
	                std::vector<std::vector<float> >& activations,
	                bool sigmoidOutput = false,
	                LayerNormParams* lnp = NULL) const;
	void dffBackward(NNetwork& net, const std::vector<std::vector<float> >& activations,
	                 const float* outputGrad, unsigned int outputSize,
	                 std::vector<float>* inputGrad,
	                 bool sigmoidOutput = false,
	                 LayerNormParams* lnp = NULL);
	void dffBackward(const NNetwork& net, const std::vector<std::vector<float> >& activations,
	                 const float* outputGrad, unsigned int outputSize,
	                 std::vector<float>* inputGrad,
	                 GradientBuffer& gradBuf,
	                 std::vector<std::vector<float> >& scratchDelta,
	                 bool sigmoidOutput = false);
	void dffBackward(const NNetwork& net, const std::vector<std::vector<float> >& activations,
	                 const float* outputGrad, unsigned int outputSize,
	                 std::vector<float>* inputGrad,
	                 GradientBuffer& gradBuf,
	                 std::vector<std::vector<float> >& scratchDelta,
	                 bool sigmoidOutput,
	                 const LayerNormParams* lnp,
	                 std::vector<std::vector<float> >& lnGGamma,
	                 std::vector<std::vector<float> >& lnGBeta);
	void dffUpdate(NNetwork& net, AdamState& state, float lr);

	// CNN forward/backward
	void cnnForward(const NNetwork& net, const float* input,
	                std::vector<float>& output,
	                bool sigmoidOutput = true,
	                std::vector<std::vector<float> >* fcActivations = NULL) const;
	void cnnBackward(NNetwork& net, const float* input,
	                 const float* outputGrad, unsigned int outputSize,
	                 std::vector<float>* inputGrad,
	                 const std::vector<float>* penultGrad = NULL,
	                 bool sigmoidOutput = true);
	void cnnBackward(const NNetwork& net, const float* input,
	                 const float* outputGrad, unsigned int outputSize,
	                 std::vector<float>* inputGrad,
	                 GradientBuffer& gradBuf,
	                 const std::vector<float>* penultGrad = NULL,
	                 bool sigmoidOutput = true);
	void cnnUpdate(NNetwork& net, AdamState& state, float lr);

	// Transposed-conv (deconv) generator
	bool initDeconvTensors(NNetwork& net, unsigned int inputDim, const DeconvConfig& cfg);
	void deconvForward(const NNetwork& net, const float* input, unsigned int inputSize,
	                   std::vector<float>& output,
	                   std::vector<std::vector<float> >* scratchOut = NULL,
	                   unsigned int noiseSeed = 0,
	                   DeconvScratchArena* arena = NULL) const;
	void deconvBackward(NNetwork& net, const std::vector<std::vector<float> >& scratch,
	                    const float* outputGrad, unsigned int outputSize,
	                    std::vector<float>* inputGrad,
	                    DeconvScratchArena* arena = NULL);
	void deconvBackward(const NNetwork& net, const std::vector<std::vector<float> >& scratch,
	                    const float* outputGrad, unsigned int outputSize,
	                    std::vector<float>* inputGrad,
	                    GradientBuffer& gradBuf,
	                    DeconvScratchArena* arena = NULL);
	void deconvUpdate(NNetwork& net, float lr);
	void zeroDeconvGrads(NNetwork& net);

	// Loss computation
	float vanillaDiscriminatorLoss(float predReal, float predFake) const;
	float vanillaGeneratorLoss(float predFake) const;
	float wganDiscriminatorLoss(float predReal, float predFake) const;
	float wganGeneratorLoss(float predFake) const;
	float lsganDiscriminatorLoss(float predReal, float predFake) const;
	float lsganGeneratorLoss(float predFake) const;
	float computeGradientPenalty(const float* real, const float* fake,
	                             unsigned int dim);
	float computeGradientPenalty(NNetwork& disc, const float* real, const float* fake,
	                             unsigned int dim);
	float computeGradientPenalty(const NNetwork& disc, const float* real, const float* fake,
	                             unsigned int dim, float epsilon,
	                             GradientBuffer& discardBuf,
	                             std::vector<std::vector<float> >& scratchDelta);

	// Noise sampling
	void sampleNoise(std::vector<float>& noise) const;

	// Unified training
	NNetworkStatus trainSingleDomain(const DataInput* realData, IGANCallbacks* cb);
	NNetworkStatus trainDualDomain(const DataInput* domainA, const DataInput* domainB, IGANCallbacks* cb);

	// InfoGAN helpers (parameterized to work on any Q-head)
	void sampleLatentCodes(std::vector<float>& catCode, std::vector<float>& contCode) const;
	void initQHead(unsigned int sharedDim, QNetworkHead& head, AdamState& adamSt);
	void initGenQHead(unsigned int inputDim, unsigned int hiddenDim,
	                  QNetworkHead& head, AdamState& adamSt);
	void qHeadForward(const QNetworkHead& head, const float* shared, unsigned int dim,
	                  std::vector<float>& qOut,
	                  std::vector<float>* hiddenPre = NULL,
	                  std::vector<float>* hiddenPost = NULL) const;
	void qHeadBackward(QNetworkHead& head, const float* shared, const float* qGrad,
	                   std::vector<float>& sharedGrad);
	void qHeadBackward(const QNetworkHead& head, const float* shared, const float* qGrad,
	                   std::vector<float>& sharedGrad,
	                   GradientBuffer& gradBuf);
	void qHeadUpdate(QNetworkHead& head, AdamState& adamSt, float lr);
	void genQHeadUpdate(QNetworkHead& head, AdamState& adamSt, float lr);
	float computeInfoLoss(const std::vector<float>& qOut,
	                      const std::vector<float>& catCode, const std::vector<float>& contCode,
	                      std::vector<float>& qGrad) const;

	// StyleGAN helpers (parameterized to work on any style set)
	void initStyleAffines(unsigned int wDim, const NNetwork& gen,
	                      std::vector<StyleAffine>& affines, AdamState& adamSt,
	                      std::vector<float>& scales, std::vector<float>& gScales);
	void dffForwardStyled(const NNetwork& net, const float* input, unsigned int inputSize,
	                      const std::vector<float>& w,
	                      const std::vector<StyleAffine>& affines,
	                      const std::vector<float>& scales,
	                      std::vector<std::vector<float> >& activations,
	                      std::vector<std::vector<float> >& xNorms,
	                      std::vector<float>& means, std::vector<float>& invStds,
	                      std::vector<std::vector<float> >& noiseVecs) const;
	void dffBackwardStyled(NNetwork& net, const std::vector<std::vector<float> >& activations,
	                       const float* outputGrad, unsigned int outputSize,
	                       const std::vector<float>& w,
	                       const std::vector<std::vector<float> >& xNorms,
	                       const std::vector<float>& means, const std::vector<float>& invStds,
	                       const std::vector<std::vector<float> >& noiseVecs,
	                       std::vector<StyleAffine>& affines,
	                       std::vector<float>& gScales,
	                       std::vector<float>& dW);
	void dffBackwardStyled(const NNetwork& net, const std::vector<std::vector<float> >& activations,
	                       const float* outputGrad, unsigned int outputSize,
	                       const std::vector<float>& w,
	                       const std::vector<std::vector<float> >& xNorms,
	                       const std::vector<float>& means, const std::vector<float>& invStds,
	                       const std::vector<std::vector<float> >& noiseVecs,
	                       const std::vector<StyleAffine>& affines,
	                       std::vector<float>& gScalesLocal,
	                       std::vector<float>& dW,
	                       GradientBuffer& gradBuf,
	                       std::vector<std::vector<float> >& styleGW,
	                       std::vector<std::vector<float> >& styleGBias,
	                       std::vector<std::vector<float> >& scratchDeltaLocal);
	void styleAffineUpdate(float lr,
	                       std::vector<StyleAffine>& affines, AdamState& adamSt,
	                       std::vector<float>& scales, std::vector<float>& gScales);

	// Layer normalization helpers
	void initLayerNorm(LayerNormParams& lnp, const NNetwork& net);
	void scaleLayerNormGrads(LayerNormParams& lnp, float scale);
	void layerNormUpdate(LayerNormParams& lnp, float lr);
	LayerNormParams* genLN() const;
	LayerNormParams* genBALN() const;

	// Zero discriminator weight gradients (utility)
	void zeroDFFGrads(NNetwork& net);
	void zeroCNNGrads(NNetwork& net);
	void zeroDiscGrads();

	// Spectral normalization (constrain Lipschitz constant of disc weights)
	void spectralNormDFF(NNetwork& net, SpectralNormState& state, int nIters);
	void spectralNormCNN(NNetwork& net, SpectralNormState& state, int nIters);

	// Sigmoid helper
	static float sigmoid(float x);
	static float sigmoidDeriv(float sigx);

	// Parallel discriminator training callback data and body
	struct DiscCritData
	{
		GAN* self;
		GANThreadCtx* ctxs;
		unsigned int nThreads;
		unsigned int curBatchSize;
		const float** realPtrs;
		const unsigned int* realSizes;
		const std::vector<float>* noise;
		const std::vector<float>* catCodes;
		const std::vector<float>* contCodes;
		const float* gpEps;
		unsigned int genInputDim;
		unsigned int dataDim;
		unsigned int numCat;
		unsigned int numCont;
		bool hasInfo;
		bool hasStyle;
		bool genIsDeconv;
		float infoLambda;
		float catScale;
		unsigned int penultDim;
		const unsigned int* beginToTid; // maps chunk begin -> thread context index
	};
	static void discCritBody(void* userData, unsigned int begin, unsigned int end);

	// Parallel generator training callback data and body
	struct GenTrainData
	{
		GAN* self;
		GANThreadCtx* ctxs;
		unsigned int nThreads;
		unsigned int curBatchSize;
		const std::vector<float>* noise;
		const std::vector<float>* catCodes;
		const std::vector<float>* contCodes;
		unsigned int genInputDim;
		unsigned int dataDim;
		unsigned int numCat;
		unsigned int numCont;
		bool hasInfo;
		bool hasStyle;
		bool genIsDeconv;
		bool hasLN;
		float infoLambda;
		float catScale;
		float divLambda;
		unsigned int penultDim;
		const unsigned int* beginToTid;
		LayerNormParams* lnScratch; // per-thread LN forward cache, or NULL
	};
	static void genTrainBody(void* userData, unsigned int begin, unsigned int end);

	// Parallel layer-wise gradient reduction callback data and body
	struct LayerReduceData
	{
		GANThreadCtx* ctxs;
		unsigned int nThreads;
		NNetwork* net;
		bool isGen;       // true=deconv generator, false=CNN discriminator
	};
	static void layerReduceBody(void* userData, unsigned int begin, unsigned int end);

	// CycleGAN parallel discriminator training callback data and body
	struct CycleDiscData
	{
		GAN* self;
		GANThreadCtx* ctxs;
		unsigned int nThreads;
		unsigned int curBatchSize;
		// Real data pointers for both domains
		const float** realPtrsA;
		const float** realPtrsB;
		// Pre-generated codes for generating fakes
		const std::vector<float>* catCodes;
		const std::vector<float>* contCodes;
		// Pre-generated style noise (only used if hasStyle)
		const std::vector<float>* styleNoise;
		// GP epsilons
		const float* gpEps;
		// Dimensions
		unsigned int genInputDim;    // input dim of the generator (srcDim + codes)
		unsigned int srcDim;         // source domain dimension (input to generator)
		unsigned int tgtDim;         // target domain dimension (real data for disc)
		unsigned int numCat;
		unsigned int numCont;
		// Flags
		bool hasInfo;
		bool hasStyle;
		bool genIsDeconv;
		float infoLambda;
		unsigned int penultDim;
		const unsigned int* beginToTid;
		// Which discriminator/generator to use (true = training disc A, false = disc B)
		bool isDiscA;
	};
	static void cycleDiscBody(void* userData, unsigned int begin, unsigned int end);

	// CycleGAN parallel generator training callback data and body
	struct CycleGenData
	{
		GAN* self;
		GANThreadCtx* ctxs;
		unsigned int nThreads;
		unsigned int curBatchSize;
		// Real data pointers
		const float** realPtrsA;
		const float** realPtrsB;
		// Pre-generated codes
		const std::vector<float>* catCodesAB;
		const std::vector<float>* contCodesAB;
		const std::vector<float>* catCodesBA;
		const std::vector<float>* contCodesBA;
		// Pre-generated style noise (if hasStyle)
		const std::vector<float>* styleNoiseAB;
		const std::vector<float>* styleNoiseBA;
		const std::vector<float>* styleNoiseRecA;
		const std::vector<float>* styleNoiseRecB;
		const std::vector<float>* styleNoiseIdentB;
		const std::vector<float>* styleNoiseIdentA;
		// Dimensions
		unsigned int genABInputDim;
		unsigned int genBAInputDim;
		unsigned int dimA;
		unsigned int dimB;
		unsigned int numCat;
		unsigned int numCont;
		// Flags
		bool hasInfo;
		bool hasStyle;
		bool genIsDeconv;
		bool hasLN;
		bool hasBALN;
		float cycleLambda;
		float identityLambda;
		float infoLambda;
		unsigned int penultDimA;
		unsigned int penultDimB;
		const unsigned int* beginToTid;
		// Per-thread LN forward scratch
		LayerNormParams* lnScratchAB;
		LayerNormParams* lnScratchBA;
	};
	static void cycleGenBody(void* userData, unsigned int begin, unsigned int end);
};

} // namespace glades

#endif
