	// === Transformer structured metrics (serving/inference) ===
	//
	// This is a lightweight, dependency-free metrics surface intended for production serving.
	// It is disabled by default. When enabled, KV-cache session Reset/Append will accumulate
	// timing/counter data and the generation APIs will emit structured log lines via the network logger.
	struct TransformerMetricsConfig
	{
		// Master switch. When false, no extra timing/counters are collected.
		bool enable;
		// If true, collect coarse per-kernel timing breakdowns inside KV append.
		// This adds overhead and should only be enabled when diagnosing performance.
		bool enableKvKernelBreakdown;
		// Emit one log line per request result in batched serving APIs.
		bool logPerRequest;
		// Emit one log line per KV append (very noisy; intended for debugging only).
		bool logPerKvAppend;
		// Collect GPU launch/transfer/sync counters in train/infer hot paths.
		bool enableGpuPerf;
		// Emit one structured summary line after GPU-backed train/test runs.
		bool logGpuTrainSummary;
		// Emit one structured summary line after GPU-backed KV append operations.
		bool logGpuInferSummary;

		TransformerMetricsConfig()
		    : enable(false),
		      enableKvKernelBreakdown(true),
		      logPerRequest(true),
		      logPerKvAppend(false),
		      enableGpuPerf(false),
		      logGpuTrainSummary(false),
		      logGpuInferSummary(false)
		{
		}
	};

	struct TransformerGpuPerfBreakdown
	{
		glades::gpu::DispatchPerfCounters counters;
		double msTotal;
		double msEmbed;
		double msPosEnc;
		double msAttention;
		double msLoss;
		double msBackward;
		double msOptimizer;

		TransformerGpuPerfBreakdown()
		    : counters(),
		      msTotal(0.0),
		      msEmbed(0.0),
		      msPosEnc(0.0),
		      msAttention(0.0),
		      msLoss(0.0),
		      msBackward(0.0),
		      msOptimizer(0.0)
		{
		}

		void reset() { *this = TransformerGpuPerfBreakdown(); }
	};

	struct TransformerKvPerfBreakdown
	{
		unsigned long long kvAppends;
		// Cache effectiveness (positional encoding caches owned by sessions).
		unsigned long long sinCacheHits;
		unsigned long long sinCacheMisses;
		unsigned long long ropeCacheHits;
		unsigned long long ropeCacheMisses;
		// NaN/Inf detection
		unsigned long long nonFiniteHiddenState;
		unsigned int lastNonFiniteLayer;
		unsigned int lastNonFinitePos;

		// Wall time (ms) across KV appends (and optionally per-kernel breakdown).
		double msTotal;
		double msEmbed;
		double msPosEnc;
		double msNorm;
		double msProjQKV;
		double msRoPE;
		double msKVStore;
		double msAttention;
		double msWo;
		double msFFN;
		double msLogits;
		TransformerGpuPerfBreakdown gpu;

		TransformerKvPerfBreakdown()
		    : kvAppends(0ULL),
		      sinCacheHits(0ULL),
		      sinCacheMisses(0ULL),
		      ropeCacheHits(0ULL),
		      ropeCacheMisses(0ULL),
		      nonFiniteHiddenState(0ULL),
		      lastNonFiniteLayer(0u),
		      lastNonFinitePos(0u),
		      msTotal(0.0),
		      msEmbed(0.0),
		      msPosEnc(0.0),
		      msNorm(0.0),
		      msProjQKV(0.0),
		      msRoPE(0.0),
		      msKVStore(0.0),
		      msAttention(0.0),
		      msWo(0.0),
		      msFFN(0.0),
		      msLogits(0.0),
		      gpu()
		{
		}

		void reset() { *this = TransformerKvPerfBreakdown(); }
	};
