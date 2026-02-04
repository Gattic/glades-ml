// Copyright 2026
//
// Unified, versioned model persistence for Glades ML.
//
// A "model package" is stored at:
//   database/models/<modelName>/{manifest.txt, nninfo.csv, weights.txt}
//
// This unifies architecture persistence (NNInfo) and parameter persistence (LayerBuilder state)
// behind a single API.

#include "network.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace {

// v1: weights.txt stored legacy Node/Edge graph weights via LayerBuilder::saveStateToFile().
// v2: weights.txt stores packed tensor weights (authoritative parameters).
static const int kModelFormatVersion = 2;

static bool mkdir_if_missing(const std::string& path)
{
	if (path.empty())
		return false;

	// 0777 is filtered by umask; matches existing project style.
	if (::mkdir(path.c_str(), 0777) == 0)
		return true;

	// Already exists is fine.
	return errno == EEXIST;
}

static bool ensure_models_dir(const std::string& modelName)
{
	// Keep behavior consistent with existing relative "database/..." paths.
	if (!mkdir_if_missing("database"))
		return false;
	if (!mkdir_if_missing("database/models"))
		return false;

	if (modelName.empty())
		return false;
	return mkdir_if_missing(std::string("database/models/") + modelName);
}

static std::string model_dir(const std::string& modelName)
{
	return std::string("database/models/") + modelName + "/";
}

struct ModelManifest
{
	std::string magic;
	int version;
	int netType;
	std::string name;
	int epochs;
	bool hasEpochs;
	uint64_t rngSeed;
	bool hasSeed;
	glades::TrainingConfig trainingConfig;
	bool hasTrainingConfig;

	ModelManifest()
	    : magic(),
	      version(-1),
	      netType(-1),
	      name(),
	      epochs(0),
	      hasEpochs(false),
	      rngSeed(0u),
	      hasSeed(false),
	      trainingConfig(),
	      hasTrainingConfig(false)
	{
	}
};

static bool read_kv_manifest(const std::string& manifestPath, ModelManifest& m)
{
	m = ModelManifest();

	std::ifstream in(manifestPath.c_str());
	if (!in)
		return false;

	std::string line;
	if (!std::getline(in, line))
		return false;
	m.magic = line;

	while (std::getline(in, line))
	{
		if (line.empty())
			continue;

		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		const std::string key = line.substr(0, eq);
		const std::string val = line.substr(eq + 1);

		if (key == "version")
			m.version = atoi(val.c_str());
		else if (key == "netType")
			m.netType = atoi(val.c_str());
		else if (key == "name")
			m.name = val;
		else if (key == "epochs")
		{
			m.epochs = atoi(val.c_str());
			m.hasEpochs = true;
		}
		else if (key == "rngSeed")
		{
			// Stored as unsigned integer in text.
			m.rngSeed = static_cast<uint64_t>(strtoull(val.c_str(), NULL, 10));
			m.hasSeed = true;
		}
		// TrainingConfig (optional; missing keys keep defaults)
		else if (key == "training.minibatchSizeOverride")
		{
			m.trainingConfig.minibatchSizeOverride = atoi(val.c_str());
			m.hasTrainingConfig = true;
		}
		else if (key == "training.tbpttWindowOverride")
		{
			m.trainingConfig.tbpttWindowOverride = atoi(val.c_str());
			m.hasTrainingConfig = true;
		}
		else if (key == "training.globalGradClipNorm")
		{
			m.trainingConfig.globalGradClipNorm = static_cast<float>(atof(val.c_str()));
			m.hasTrainingConfig = true;
		}
		else if (key == "training.perElementGradClip")
		{
			m.trainingConfig.perElementGradClip = static_cast<float>(atof(val.c_str()));
			m.hasTrainingConfig = true;
		}
		else if (key == "training.lrSchedule.type")
		{
			m.trainingConfig.lrSchedule.type =
			    static_cast<glades::LearningRateScheduleConfig::Type>(atoi(val.c_str()));
			m.hasTrainingConfig = true;
		}
		else if (key == "training.lrSchedule.stepSizeEpochs")
		{
			m.trainingConfig.lrSchedule.stepSizeEpochs = atoi(val.c_str());
			m.hasTrainingConfig = true;
		}
		else if (key == "training.lrSchedule.gamma")
		{
			m.trainingConfig.lrSchedule.gamma = static_cast<float>(atof(val.c_str()));
			m.hasTrainingConfig = true;
		}
		else if (key == "training.lrSchedule.cosineTMaxEpochs")
		{
			m.trainingConfig.lrSchedule.cosineTMaxEpochs = atoi(val.c_str());
			m.hasTrainingConfig = true;
		}
		else if (key == "training.lrSchedule.minMultiplier")
		{
			m.trainingConfig.lrSchedule.minMultiplier = static_cast<float>(atof(val.c_str()));
			m.hasTrainingConfig = true;
		}
	}

	return true;
}

} // namespace

namespace glades {

NNetworkStatus NNetwork::saveModel(const std::string& modelName) const
{
	if (modelName.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveModel: modelName is empty");
	if (!skeleton)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveModel: skeleton is null");

	// Tensor-first: weights are persisted from packed tensors (the single source of truth).
	// We require a built graph only for shape/bootstrap (tensors can be initialized from it once).
	if (meat.getLayersSize() == 0 || meat.getLayerSize(0) == 0)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveModel: network graph is not built (meat is empty)");

	if (!ensure_models_dir(modelName))
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to create database/models directory");

	const std::string dir = model_dir(modelName);
	const std::string manifestPath = dir + "manifest.txt";
	const std::string nninfoPath = dir + "nninfo.csv";
	const std::string weightsPath = dir + "weights.txt";

	// 1) manifest
	{
		std::ofstream out(manifestPath.c_str());
		if (!out)
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to write manifest.txt");

		out << "GLADES_MODEL" << "\n";
		out << "version=" << kModelFormatVersion << "\n";
		out << "name=" << modelName << "\n";
		out << "netType=" << netType << "\n";
		out << "epochs=" << epochs << "\n";
		out << "rngSeed=" << static_cast<unsigned long long>(rngSeed) << "\n";
		// Persist TrainingConfig so runs are reproducible and resumable (modulo optimizer state).
		out << "training.minibatchSizeOverride=" << trainingConfig.minibatchSizeOverride << "\n";
		out << "training.tbpttWindowOverride=" << trainingConfig.tbpttWindowOverride << "\n";
		out << "training.globalGradClipNorm=" << trainingConfig.globalGradClipNorm << "\n";
		out << "training.perElementGradClip=" << trainingConfig.perElementGradClip << "\n";
		out << "training.lrSchedule.type=" << static_cast<int>(trainingConfig.lrSchedule.type) << "\n";
		out << "training.lrSchedule.stepSizeEpochs=" << trainingConfig.lrSchedule.stepSizeEpochs << "\n";
		out << "training.lrSchedule.gamma=" << trainingConfig.lrSchedule.gamma << "\n";
		out << "training.lrSchedule.cosineTMaxEpochs=" << trainingConfig.lrSchedule.cosineTMaxEpochs << "\n";
		out << "training.lrSchedule.minMultiplier=" << trainingConfig.lrSchedule.minMultiplier << "\n";
	}

	// 2) architecture
	{
		// NNInfo::toGTable() is private; NNetwork is a friend of NNInfo.
		const shmea::GTable t = skeleton->toGTable();
		t.save(shmea::GString(nninfoPath.c_str()));
	}

	// 3) weights
	{
		const NNetworkStatus stW = saveTensorWeightsToFile(weightsPath);
		if (!stW.ok())
			return stW;
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

NNetworkStatus NNetwork::loadModel(const std::string& modelName, const DataInput* forShape, int netTypeOverride)
{
	if (modelName.empty())
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadModel: modelName is empty");
	if (!forShape)
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadModel: forShape is null (required to rebuild input layer)");

	const std::string dir = model_dir(modelName);
	const std::string manifestPath = dir + "manifest.txt";
	const std::string nninfoPath = dir + "nninfo.csv";
	const std::string weightsPath = dir + "weights.txt";

	// Prefer the unified format. If it's not present, fall back to legacy persistence.
	{
		std::ifstream probe(manifestPath.c_str());
		if (!probe)
		{
			// Legacy:
			// - architecture in `database/neuralnetworks/<name>`
			// - weights in `database/nn-state/<name>`
			if (!load(shmea::GString(modelName.c_str())))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: unable to load legacy skeleton (NNInfo)");

			if (netTypeOverride >= 0)
				netType = netTypeOverride;

			if (!meat.build(skeleton, forShape, netType, false))
				return failStatus(NNetworkStatus::BUILD_FAILED, std::string("loadModel: build failed: ") + meat.getLastError());

			mustBuildMeat = false;

			if (!meat.loadState(skeleton, modelName.c_str()))
				return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: unable to load legacy weights (nn-state)");

			// Legacy persistence loads weights into the Node/Edge graph. Bootstrap tensors from it.
			if (!ensureTensorParametersInitializedFromGraph())
				return lastStatus;
			graphWeightsDirty = false;

			return NNetworkStatus(NNetworkStatus::OK, std::string());
		}
	}

	// Unified package load.
	ModelManifest mf;
	if (!read_kv_manifest(manifestPath, mf))
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: unable to read manifest.txt");

	if (mf.magic != "GLADES_MODEL")
		return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest magic mismatch");

	if (mf.version != 1 && mf.version != kModelFormatVersion)
	{
		std::ostringstream oss;
		oss << "loadModel: unsupported model format version " << mf.version << " (expected 1 or " << kModelFormatVersion << ")";
		return failStatus(NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedNetType = mf.netType;
	if (savedNetType < 0)
		savedNetType = TYPE_DFF;

	if (netTypeOverride >= 0)
		netType = netTypeOverride;
	else
		netType = savedNetType;

	// Load architecture from nninfo.csv (self-contained, file-based).
	{
		const shmea::GTable t(shmea::GString(nninfoPath.c_str()), ',', shmea::GTable::TYPE_FILE);
		ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(shmea::GString(modelName.c_str()), t));
		skeleton = ownedSkeleton.get();
	}

	if (!skeleton)
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: failed to construct NNInfo from nninfo.csv");

	// Build weights graph for this dataset shape, then apply weights.
	if (!meat.build(skeleton, forShape, netType, false))
		return failStatus(NNetworkStatus::BUILD_FAILED, std::string("loadModel: build failed: ") + meat.getLastError());

	mustBuildMeat = false;

	// v1 stored Node/Edge weights; v2 stores packed tensor weights.
	if (mf.version == 1)
	{
		if (!meat.loadStateFromFile(skeleton, weightsPath))
			return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: unable to load v1 weights.txt (graph format)");

		// Bootstrap tensors from loaded graph so tensors become the single source of truth.
		if (!ensureTensorParametersInitializedFromGraph())
			return lastStatus;
		graphWeightsDirty = false;
	}
	else
	{
		const NNetworkStatus stW = loadTensorWeightsFromFile(weightsPath);
		if (!stW.ok())
			return stW;
		// Graph is now a stale debug view until explicitly materialized.
		graphWeightsDirty = true;
	}

	// Restore metadata/config from manifest (best-effort; missing keys keep defaults).
	if (mf.hasEpochs)
		epochs = mf.epochs;
	if (mf.hasSeed)
		setSeed(mf.rngSeed);
	if (mf.hasTrainingConfig)
		trainingConfig = mf.trainingConfig;

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

} // namespace glades

