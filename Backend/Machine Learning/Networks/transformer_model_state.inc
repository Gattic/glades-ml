	// === Transformer (encoder/decoder) packed parameters ===
	//
	// Transformer models are sequence models, like recurrent nets, but without recurrence.
	// We store all parameters in packed vectors and run explicit forward/backward kernels.
	struct TensorTransformerState
	{
		bool initialized;
		// Dataset shapes
		unsigned int inputSize; // featureCount per timestep
		unsigned int outSize;   // expected output size per timestep

		// Model config
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		// Grouped-query attention: number of KV heads (<= nHeads).
		// If equal to nHeads, this is standard multi-head attention.
		unsigned int nKVHeads;
		unsigned int nLayers;
		// If true, use causal self-attention (decoder-only / autoregressive).
		bool causal;
		// FFN kind (see TransformerRunConfig::FFNKind). Stored for shape consistency.
		unsigned int ffnKind;

		// Language-model (token) mode: embedding + vocab head.
		bool tokenModel;
		unsigned int vocabSize;
		int padTokenId;
		bool tieEmbeddings;
		// Optimizer update step (used for Adam bias correction).
		unsigned long long optimizerStep;
		// Token embedding table E: [vocabSize, dModel]
		std::vector<float> tokE;
		// Low-precision weight copy of tokE used by mixed-precision training (optional).
		std::vector<uint16_t> tokELowp;
		std::vector<float> vTokE;
		std::vector<float> v2TokE;
		std::vector<float> gTokE;
		// LM head bias: [vocabSize]
		std::vector<float> lmBias;
		std::vector<float> mLmBias;
		std::vector<float> v2LmBias;
		std::vector<float> gLmBias;

		// Input projection: x[t,inputSize] -> h[t,dModel]
		// WIn: [dModel, inputSize]
		std::vector<float> WIn;
		// Low-precision weight copy of WIn used by mixed-precision training (optional).
		std::vector<uint16_t> WInLowp;
		std::vector<float> vWIn;
		std::vector<float> v2WIn;
		std::vector<float> gWIn;
		std::vector<float> bIn;   // [dModel]
		std::vector<float> mBIn;
		std::vector<float> v2BIn;
		std::vector<float> gBIn;

		// Output projection: h[t,dModel] -> y[t,outSize]
		// WOut: [outSize, dModel]
		std::vector<float> WOut;
		// Low-precision weight copy of WOut used by mixed-precision training (optional).
		std::vector<uint16_t> WOutLowp;
		std::vector<float> vWOut;
		std::vector<float> v2WOut;
		std::vector<float> gWOut;
		std::vector<float> bOut;  // [outSize]
		std::vector<float> mBOut;
		std::vector<float> v2BOut;
		std::vector<float> gBOut;

		struct Block
		{
			// Pre-LN 1
			std::vector<float> ln1Gamma; // [dModel]
			std::vector<float> ln1Beta;  // [dModel]
			std::vector<float> mLn1Gamma;
			std::vector<float> v2Ln1Gamma;
			std::vector<float> mLn1Beta;
			std::vector<float> v2Ln1Beta;
			std::vector<float> gLn1Gamma;
			std::vector<float> gLn1Beta;

			// Self-attention linear projections (packed as [dModel, dModel])
			std::vector<float> Wq, Wk, Wv, Wo;
			// Low-precision copies (optional; used when TrainingConfig::mixedPrecision.enable).
			std::vector<uint16_t> WqLowp, WkLowp, WvLowp, WoLowp;
			std::vector<float> vWq, vWk, vWv, vWo;
			std::vector<float> v2Wq, v2Wk, v2Wv, v2Wo;
			std::vector<float> gWq, gWk, gWv, gWo;
			std::vector<float> bq, bk, bv, bo; // [dModel]
			std::vector<float> mBq, mBk, mBv, mBo;
			std::vector<float> v2Bq, v2Bk, v2Bv, v2Bo;
			std::vector<float> gBq, gBk, gBv, gBo;

			// Pre-LN 2
			std::vector<float> ln2Gamma; // [dModel]
			std::vector<float> ln2Beta;  // [dModel]
			std::vector<float> mLn2Gamma;
			std::vector<float> v2Ln2Gamma;
			std::vector<float> mLn2Beta;
			std::vector<float> v2Ln2Beta;
			std::vector<float> gLn2Gamma;
			std::vector<float> gLn2Beta;

			// Feed-forward network
			// W1: [dFF, dModel], b1: [dFF]
			// W2: [dModel, dFF], b2: [dModel]
			std::vector<float> W1, W2;
			// Low-precision copies (optional; used when TrainingConfig::mixedPrecision.enable).
			std::vector<uint16_t> W1Lowp, W2Lowp;
			std::vector<float> vW1, vW2;
			std::vector<float> v2W1, v2W2;
			std::vector<float> gW1, gW2;
			std::vector<float> b1, b2;
			std::vector<float> mB1, mB2;
			std::vector<float> v2B1, v2B2;
			std::vector<float> gB1, gB2;

			// ATLAS optimizer state per weight matrix in this block.
			atlas::WeightState atlasWq, atlasWk, atlasWv, atlasWo;
			atlas::WeightState atlasW1, atlasW2;
			atlas::EchoWeightState echoWq, echoWk, echoWv, echoWo;
			atlas::EchoWeightState echoW1, echoW2;
			atlas::BiMAPWeightState bimapWq, bimapWk, bimapWv, bimapWo;
			atlas::BiMAPWeightState bimapW1, bimapW2;
			atlas::BiMAPWeightState pactWq, pactWk, pactWv, pactWo;
			atlas::BiMAPWeightState pactW1, pactW2;
			atlas::BiMAPWeightState racerWq, racerWk, racerWv, racerWo;
			atlas::BiMAPWeightState racerW1, racerW2;
			atlas::KronWeightState kronWq, kronWk, kronWv, kronWo;
			atlas::KronWeightState kronW1, kronW2;
			atlas::MatraWeightState matraWq, matraWk, matraWv, matraWo;
			atlas::MatraWeightState matraW1, matraW2;
			atlas::ArgosWeightState argosWq, argosWk, argosWv, argosWo;
			atlas::ArgosWeightState argosW1, argosW2;
			atlas::MuonWeightState muonWq, muonWk, muonWv, muonWo;
			atlas::MuonWeightState muonW1, muonW2;
		};

		std::vector<Block> blocks;

		// ATLAS optimizer state for non-block weight matrices.
		atlas::WeightState atlasWIn;
		atlas::WeightState atlasWOut;
		atlas::WeightState atlasTokE;
		atlas::EchoWeightState echoWIn;
		atlas::EchoWeightState echoWOut;
		atlas::EchoWeightState echoTokE;
		atlas::BiMAPWeightState bimapWIn;
		atlas::BiMAPWeightState bimapWOut;
		atlas::BiMAPWeightState bimapTokE;
		atlas::BiMAPWeightState pactWIn;
		atlas::BiMAPWeightState pactWOut;
		atlas::BiMAPWeightState pactTokE;
		atlas::BiMAPWeightState racerWIn;
		atlas::BiMAPWeightState racerWOut;
		atlas::BiMAPWeightState racerTokE;
		atlas::KronWeightState kronWIn;
		atlas::KronWeightState kronWOut;
		atlas::KronWeightState kronTokE;
		atlas::MatraWeightState matraWIn;
		atlas::MatraWeightState matraWOut;
		atlas::MatraWeightState matraTokE;
		atlas::ArgosWeightState argosWIn;
		atlas::ArgosWeightState argosWOut;
		atlas::ArgosWeightState argosTokE;
		atlas::MuonWeightState muonWIn;
		atlas::MuonWeightState muonWOut;
		atlas::MuonWeightState muonTokE;

		// Benchmark-only grouped optimizer-gap diagnostics. These stay at the
		// top level so AdamW and plain ATLAS-BSRP runs can report the same
		// transformer profile without depending on ASTER state.
		unsigned long long gapApplyCount;
		double gapInputUpdateNormSum;
		std::vector<double> gapBlockUpdateNormSums;
		double gapFinalNormUpdateNormSum;
		double gapHeadUpdateNormSum;
		double gapHeadShareSum;
		double gapNonHeadShareSum;
		double gapApplyNsSum;
		unsigned long long gapMarginSnapshotCount;
		double gapTargetMarginSum;
		double gapHardNegativeLogitSum;
		std::vector<float> adamGroupPrevStepRms;
		std::vector<float> adamGroupLastScale;

		struct HelmState
		{
			bool initialized;
			unsigned int hiddenDim;
			unsigned int outputDim;
			unsigned int hiddenStackDepth;
			unsigned int modeRank;
			std::vector<unsigned int> trackedBlockIndices;
			std::vector<float> prevHiddenMean;
			std::vector<float> prevResidualMean;
			std::vector<float> hiddenVar;
			std::vector<float> residualVar;
			std::vector<float> crossCov;
			std::vector<float> sigma;
			std::vector<float> leftMode;
			std::vector<float> rightMode;
			std::vector<float> batchHiddenSum;
			std::vector<float> batchHiddenSqSum;
			std::vector<float> batchResidualSum;
			std::vector<float> batchResidualSqSum;
			std::vector<float> latent;
			std::vector<float> poleNumer;
			std::vector<float> poleDenom;
			std::vector<float> pole;
			std::vector<float> forwardCorrection;
			unsigned int batchTokenCount;
			unsigned int lastActiveModes;
			float lastEdge;
			float lastSecondEdge;
			float lastSecondEdgeRatio;
			float lastSigma;
			float lastPredR2;
			float lastMemoryGain;

			HelmState()
			    : initialized(false),
			      hiddenDim(0u),
			      outputDim(0u),
			      hiddenStackDepth(0u),
			      modeRank(0u),
			      batchTokenCount(0u),
			      lastActiveModes(0u),
			      lastEdge(0.0f),
			      lastSecondEdge(0.0f),
			      lastSecondEdgeRatio(0.0f),
			      lastSigma(0.0f),
			      lastPredR2(0.0f),
			      lastMemoryGain(0.0f)
			{
			}

			void reset()
			{
				initialized = false;
				hiddenDim = 0u;
				outputDim = 0u;
				hiddenStackDepth = 0u;
				modeRank = 0u;
				trackedBlockIndices.clear();
				prevHiddenMean.clear();
				prevResidualMean.clear();
				hiddenVar.clear();
				residualVar.clear();
				crossCov.clear();
				sigma.clear();
				leftMode.clear();
				rightMode.clear();
				batchHiddenSum.clear();
				batchHiddenSqSum.clear();
				batchResidualSum.clear();
				batchResidualSqSum.clear();
				latent.clear();
				poleNumer.clear();
				poleDenom.clear();
				pole.clear();
				forwardCorrection.clear();
				batchTokenCount = 0u;
				lastActiveModes = 0u;
				lastEdge = 0.0f;
				lastSecondEdge = 0.0f;
				lastSecondEdgeRatio = 0.0f;
				lastSigma = 0.0f;
				lastPredR2 = 0.0f;
				lastMemoryGain = 0.0f;
			}
		};

		struct AsterState
		{
			bool initialized;
			unsigned int regimeCount;
			unsigned int sketchDim;
			unsigned int supportDim;
			unsigned int tokenCondDim;
			unsigned int kappaObsDim;
			unsigned int kappaHeads;
			unsigned int kappaLagBuckets;
			unsigned int kappaRank;
			unsigned int controlDim;
			unsigned int hiddenStackDepth;
			unsigned int stateRank;
			std::vector<unsigned int> trackedBlockIndices;
			std::vector<float> prevPrevControlMean;
			std::vector<float> prevControlMean;
			std::vector<float> prevPrevResidualMean;
			std::vector<float> prevResidualMean;
			std::vector<float> controlVar;
			std::vector<float> residualVar;
			std::vector<float> transportPastCov;
			std::vector<float> transportCrossCov;
			std::vector<float> pastCov;
			std::vector<float> crossCov;
			std::vector<float> theta;
			std::vector<float> statePastCov;
			std::vector<float> stateCrossCov;
			std::vector<float> innovationCov;
			std::vector<float> innovationCross;
			std::vector<float> sigma;
			std::vector<float> leftMode;
			std::vector<float> rightMode;
			std::vector<float> batchFinalHiddenRawSum;
			std::vector<float> batchFinalHiddenSketchSum;
			std::vector<float> batchLayerHiddenRawSum;
			std::vector<float> batchLayerHiddenSketchSum;
			std::vector<float> batchLayerAttnRawSum;
			std::vector<float> batchLayerAttnSketchSum;
			std::vector<float> batchLayerPatternSum;
			std::vector<float> batchLayerKappaSum;
			std::vector<float> batchResidualSum;
			std::vector<float> batchSupportLogitSum;
			std::vector<float> batchSupportResidualSum;
			std::vector<float> batchSupportCount;
			std::vector<float> batchSupportHiddenRawSum;
			std::vector<float> batchTargetMarginSum;
			std::vector<float> batchHardNegativeLogitSum;
			std::vector<float> batchBaselineWorseSum;
			std::vector<unsigned int> batchTouchedIds;
			std::vector<unsigned int> batchSupportIds;
			std::vector<float> batchSupportRoleCounts;
			std::vector<float> batchRegimeTokenCount;
			unsigned int batchTokenCount;
			std::vector<float> targetMarginEma;
			std::vector<float> hardNegativeLogitEma;
			std::vector<float> hardMarginShortfallEma;
			std::vector<float> prevPrevLatent;
			std::vector<float> latent;
			std::vector<float> poleNumer;
			std::vector<float> poleDenom;
			std::vector<float> pole;
			unsigned int lastActiveModes;
			float lastEdge;
			float lastSecondEdge;
			float lastSecondEdgeRatio;
			float lastSigma;
			float lastPredR2;
			float lastMemoryGain;
			float lastPoleSummary;
			float aegisPredictiveErrorEma;
			float aegisOutputErrorEma;
			float aegisPrevPredictiveScore;
			float aegisPrevOutputScore;
			float aegisLastLambdaSpatial;
			float aegisLastLambdaPredictive;
			float aegisLastLambdaOutput;
			float aegisLastPredictivePredicted;
			float aegisLastPredictiveRealized;
			float aegisLastOutputPredicted;
			float aegisLastOutputRealized;
			float aegisLastChannelDisagreement;
			float citadelPredictiveTrustEma;
			float citadelOutputTrustEma;
			float citadelLastAnchor;
			float citadelLastHardRegimeMass;
			float citadelLastSparrowTrust;
			float rampartLastTau;
			float rampartLastBudget;
			float rampartLastCovariance;
			float rampartLastSparrowTrust;
			float meritLastTau;
			float meritLastBudget;
			float meritLastCovariance;
			float meritLastSparrowTrust;
			float meritLastGeometryTrust;
			float strataLastNullMode;
			float strataLastPredictiveMode;
			float strataLastOutputMode;
			float strataLastCoupledMode;
			float strataLastBudget;
			float strataNullBenefitEma;
			float strataPredictiveBenefitEma;
			float strataOutputBenefitEma;
			float strataCoupledBenefitEma;
			float strataLastNullBenefit;
			float strataLastPredictiveBenefit;
			float strataLastOutputBenefit;
			float strataLastCoupledBenefit;
			float strataLastSelectedExcess;
			float strataLastSwitchRate;
			unsigned long long timingBoundaryCount;
			double totalBoundaryNs;
			double totalSetupNs;
			double totalTransportNs;
			double totalTransferFitNs;
			double totalStateFitNs;
			double totalInnovationFitNs;
			double totalApplyNs;

			AsterState()
			    : initialized(false),
			      regimeCount(0u),
			      sketchDim(0u),
			      supportDim(0u),
			      tokenCondDim(0u),
			      kappaObsDim(0u),
			      kappaHeads(0u),
			      kappaLagBuckets(0u),
			      kappaRank(0u),
			      controlDim(0u),
			      hiddenStackDepth(0u),
			      stateRank(0u),
			      batchTokenCount(0u),
			      lastActiveModes(0u),
			      lastEdge(0.0f),
			      lastSecondEdge(0.0f),
			      lastSecondEdgeRatio(0.0f),
			      lastSigma(0.0f),
			      lastPredR2(0.0f),
			      lastMemoryGain(0.0f),
			      lastPoleSummary(0.0f),
			      aegisPredictiveErrorEma(0.0f),
			      aegisOutputErrorEma(0.0f),
			      aegisPrevPredictiveScore(0.0f),
			      aegisPrevOutputScore(0.0f),
			      aegisLastLambdaSpatial(0.0f),
			      aegisLastLambdaPredictive(0.0f),
			      aegisLastLambdaOutput(0.0f),
			      aegisLastPredictivePredicted(0.0f),
			      aegisLastPredictiveRealized(0.0f),
			      aegisLastOutputPredicted(0.0f),
			      aegisLastOutputRealized(0.0f),
			      aegisLastChannelDisagreement(0.0f),
			      citadelPredictiveTrustEma(0.0f),
			      citadelOutputTrustEma(0.0f),
			      citadelLastAnchor(0.0f),
			      citadelLastHardRegimeMass(0.0f),
			      citadelLastSparrowTrust(1.0f),
			      rampartLastTau(0.0f),
			      rampartLastBudget(0.0f),
			      rampartLastCovariance(0.0f),
			      rampartLastSparrowTrust(1.0f),
			      meritLastTau(0.0f),
			      meritLastBudget(0.0f),
			      meritLastCovariance(0.0f),
			      meritLastSparrowTrust(1.0f),
			      meritLastGeometryTrust(0.0f),
			      strataLastNullMode(1.0f),
			      strataLastPredictiveMode(0.0f),
			      strataLastOutputMode(0.0f),
			      strataLastCoupledMode(0.0f),
			      strataLastBudget(0.0f),
			      strataNullBenefitEma(0.0f),
			      strataPredictiveBenefitEma(0.0f),
			      strataOutputBenefitEma(0.0f),
			      strataCoupledBenefitEma(0.0f),
			      strataLastNullBenefit(0.0f),
			      strataLastPredictiveBenefit(0.0f),
			      strataLastOutputBenefit(0.0f),
			      strataLastCoupledBenefit(0.0f),
			      strataLastSelectedExcess(0.0f),
			      strataLastSwitchRate(0.0f),
			      timingBoundaryCount(0ULL),
			      totalBoundaryNs(0.0),
			      totalSetupNs(0.0),
			      totalTransportNs(0.0),
			      totalTransferFitNs(0.0),
			      totalStateFitNs(0.0),
			      totalInnovationFitNs(0.0),
			      totalApplyNs(0.0)
			{
			}

			void reset()
			{
				initialized = false;
				regimeCount = 0u;
				sketchDim = 0u;
				supportDim = 0u;
				tokenCondDim = 0u;
				kappaObsDim = 0u;
				kappaHeads = 0u;
				kappaLagBuckets = 0u;
				kappaRank = 0u;
				controlDim = 0u;
				hiddenStackDepth = 0u;
				stateRank = 0u;
				trackedBlockIndices.clear();
				prevPrevControlMean.clear();
				prevControlMean.clear();
				prevPrevResidualMean.clear();
				prevResidualMean.clear();
				controlVar.clear();
				residualVar.clear();
				transportPastCov.clear();
				transportCrossCov.clear();
				pastCov.clear();
				crossCov.clear();
				theta.clear();
				statePastCov.clear();
				stateCrossCov.clear();
				innovationCov.clear();
				innovationCross.clear();
				sigma.clear();
				leftMode.clear();
				rightMode.clear();
				batchFinalHiddenRawSum.clear();
				batchFinalHiddenSketchSum.clear();
				batchLayerHiddenRawSum.clear();
				batchLayerHiddenSketchSum.clear();
				batchLayerAttnRawSum.clear();
				batchLayerAttnSketchSum.clear();
				batchLayerPatternSum.clear();
				batchLayerKappaSum.clear();
				batchResidualSum.clear();
				batchSupportLogitSum.clear();
				batchSupportResidualSum.clear();
				batchSupportCount.clear();
				batchSupportHiddenRawSum.clear();
				batchTargetMarginSum.clear();
				batchHardNegativeLogitSum.clear();
				batchBaselineWorseSum.clear();
				batchTouchedIds.clear();
				batchSupportIds.clear();
				batchSupportRoleCounts.clear();
				batchRegimeTokenCount.clear();
				batchTokenCount = 0u;
				targetMarginEma.clear();
				hardNegativeLogitEma.clear();
				hardMarginShortfallEma.clear();
				prevPrevLatent.clear();
				latent.clear();
				poleNumer.clear();
				poleDenom.clear();
				pole.clear();
				lastActiveModes = 0u;
				lastEdge = 0.0f;
				lastSecondEdge = 0.0f;
				lastSecondEdgeRatio = 0.0f;
				lastSigma = 0.0f;
				lastPredR2 = 0.0f;
				lastMemoryGain = 0.0f;
				lastPoleSummary = 0.0f;
				aegisPredictiveErrorEma = 0.0f;
				aegisOutputErrorEma = 0.0f;
				aegisPrevPredictiveScore = 0.0f;
				aegisPrevOutputScore = 0.0f;
				aegisLastLambdaSpatial = 0.0f;
				aegisLastLambdaPredictive = 0.0f;
				aegisLastLambdaOutput = 0.0f;
				aegisLastPredictivePredicted = 0.0f;
				aegisLastPredictiveRealized = 0.0f;
				aegisLastOutputPredicted = 0.0f;
				aegisLastOutputRealized = 0.0f;
				aegisLastChannelDisagreement = 0.0f;
				citadelPredictiveTrustEma = 0.0f;
				citadelOutputTrustEma = 0.0f;
				citadelLastAnchor = 0.0f;
				citadelLastHardRegimeMass = 0.0f;
				citadelLastSparrowTrust = 1.0f;
				rampartLastTau = 0.0f;
				rampartLastBudget = 0.0f;
				rampartLastCovariance = 0.0f;
				rampartLastSparrowTrust = 1.0f;
				meritLastTau = 0.0f;
				meritLastBudget = 0.0f;
				meritLastCovariance = 0.0f;
				meritLastSparrowTrust = 1.0f;
				meritLastGeometryTrust = 0.0f;
				strataLastNullMode = 1.0f;
				strataLastPredictiveMode = 0.0f;
				strataLastOutputMode = 0.0f;
				strataLastCoupledMode = 0.0f;
				strataLastBudget = 0.0f;
				strataNullBenefitEma = 0.0f;
				strataPredictiveBenefitEma = 0.0f;
				strataOutputBenefitEma = 0.0f;
				strataCoupledBenefitEma = 0.0f;
				strataLastNullBenefit = 0.0f;
				strataLastPredictiveBenefit = 0.0f;
				strataLastOutputBenefit = 0.0f;
				strataLastCoupledBenefit = 0.0f;
				strataLastSelectedExcess = 0.0f;
				strataLastSwitchRate = 0.0f;
				timingBoundaryCount = 0ULL;
				totalBoundaryNs = 0.0;
				totalSetupNs = 0.0;
				totalTransportNs = 0.0;
				totalTransferFitNs = 0.0;
				totalStateFitNs = 0.0;
				totalInnovationFitNs = 0.0;
				totalApplyNs = 0.0;
			}
		};

		// Final LayerNorm (applied after the last block, before the output head).
		std::vector<float> lnFinalGamma; // [dModel]
		std::vector<float> lnFinalBeta;  // [dModel]
		std::vector<float> mLnFinalGamma; // Adam 1st moment
		std::vector<float> v2LnFinalGamma; // Adam 2nd moment
		std::vector<float> mLnFinalBeta;
		std::vector<float> v2LnFinalBeta;
		std::vector<float> gLnFinalGamma; // gradients
		std::vector<float> gLnFinalBeta;
		// Running accumulators for Adam bias correction (avoids pow(beta,t) each step).
		double adamBeta1Power;
		double adamBeta2Power;

		// === Mixed precision runtime state (Transformer training) ===
		//
		// Master weights remain the FP32 vectors above.
		// If mixed precision is enabled, the training path uses the low-precision weight copies
		// (tokELowp/WInLowp/WOutLowp + per-block lowp matrices) for forward/backward GEMMs/GEMVs.
		bool mpLowpReady;
		// Low-precision dtype selector (transformer_kernels::LOWP_F16 or transformer_kernels::LOWP_BF16).
		int mpLowpDType;
		// Dynamic loss scaling state.
		float mpLossScale;
		int mpLossScaleGoodSteps;
		HelmState helm;
		AsterState aster;

		TensorTransformerState()
		    : initialized(false),
		      inputSize(0u),
		      outSize(0u),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      causal(false)
		      ,
		      ffnKind(0u)
		      ,
		      tokenModel(false),
		      vocabSize(0u),
		      padTokenId(-1),
		      tieEmbeddings(true),
		      optimizerStep(0ULL),
		      gapApplyCount(0ULL),
		      gapInputUpdateNormSum(0.0),
		      gapBlockUpdateNormSums(),
		      gapFinalNormUpdateNormSum(0.0),
		      gapHeadUpdateNormSum(0.0),
		      gapHeadShareSum(0.0),
		      gapNonHeadShareSum(0.0),
		      gapApplyNsSum(0.0),
		      gapMarginSnapshotCount(0ULL),
		      gapTargetMarginSum(0.0),
		      gapHardNegativeLogitSum(0.0),
		      adamBeta1Power(1.0),
		      adamBeta2Power(1.0),
		      mpLowpReady(false),
		      mpLowpDType(0),
		      mpLossScale(1.0f),
		      mpLossScaleGoodSteps(0)
		{
		}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			dModel = 0u;
			dFF = 0u;
			nHeads = 0u;
			nKVHeads = 0u;
			nLayers = 0u;
			causal = false;
			ffnKind = 0u;
			tokenModel = false;
			vocabSize = 0u;
			padTokenId = -1;
			tieEmbeddings = true;
			optimizerStep = 0ULL;
			gapApplyCount = 0ULL;
			gapInputUpdateNormSum = 0.0;
			gapBlockUpdateNormSums.clear();
			gapFinalNormUpdateNormSum = 0.0;
			gapHeadUpdateNormSum = 0.0;
			gapHeadShareSum = 0.0;
			gapNonHeadShareSum = 0.0;
			gapApplyNsSum = 0.0;
			gapMarginSnapshotCount = 0ULL;
			gapTargetMarginSum = 0.0;
			gapHardNegativeLogitSum = 0.0;
			adamGroupPrevStepRms.clear();
			adamGroupLastScale.clear();
			adamBeta1Power = 1.0;
			adamBeta2Power = 1.0;
			mpLowpReady = false;
			mpLowpDType = 0;
			mpLossScale = 1.0f;
			mpLossScaleGoodSteps = 0;
			lnFinalGamma.clear(); lnFinalBeta.clear();
			mLnFinalGamma.clear(); v2LnFinalGamma.clear();
			mLnFinalBeta.clear(); v2LnFinalBeta.clear();
			gLnFinalGamma.clear(); gLnFinalBeta.clear();
			tokE.clear(); tokELowp.clear(); vTokE.clear(); v2TokE.clear(); gTokE.clear();
			lmBias.clear(); mLmBias.clear(); v2LmBias.clear(); gLmBias.clear();
			WIn.clear(); WInLowp.clear(); vWIn.clear(); v2WIn.clear(); gWIn.clear();
			bIn.clear(); mBIn.clear(); v2BIn.clear(); gBIn.clear();
			WOut.clear(); WOutLowp.clear(); vWOut.clear(); v2WOut.clear(); gWOut.clear();
			bOut.clear(); mBOut.clear(); v2BOut.clear(); gBOut.clear();
			blocks.clear();
			helm.reset();
			aster.reset();
		}
	};
