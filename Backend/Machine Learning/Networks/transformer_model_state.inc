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
		};

		std::vector<Block> blocks;

		// ATLAS optimizer state for non-block weight matrices.
		atlas::WeightState atlasWIn;
		atlas::WeightState atlasWOut;
		atlas::WeightState atlasTokE;

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
		}
	};
