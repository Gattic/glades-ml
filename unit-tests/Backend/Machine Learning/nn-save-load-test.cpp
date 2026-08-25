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
#include "nn-save-load-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"

#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {
static bool read_file_to_string(const std::string& path, std::string& out)
{
	out.clear();
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	std::string s;
	char buf[4096];
	while (in.good())
	{
		in.read(buf, sizeof(buf));
		const std::streamsize n = in.gcount();
		if (n > 0)
			s.append(buf, static_cast<size_t>(n));
	}
	out.swap(s);
	return true;
}

static bool mkdir_if_missing(const std::string& path)
{
	if (path.empty())
		return false;
	// 0777 filtered by umask (matches project style).
	if (::mkdir(path.c_str(), 0777) == 0)
		return true;
	// EEXIST is ok; but we avoid errno include in unit-tests.
	return true;
}

static bool write_text_file(const std::string& path, const std::string& content)
{
	std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
	if (!out)
		return false;
	out.write(content.c_str(), static_cast<std::streamsize>(content.size()));
	return static_cast<bool>(out);
}

static bool replace_line_prefix_in_file(const std::string& path, const std::string& prefix, const std::string& newLine)
{
	std::string s;
	if (!read_file_to_string(path, s))
		return false;
	std::string out;
	out.reserve(s.size() + 32u);
	size_t pos = 0;
	bool replaced = false;
	while (pos < s.size())
	{
		const size_t nl = s.find('\n', pos);
		const size_t end = (nl == std::string::npos) ? s.size() : nl;
		const std::string line = s.substr(pos, end - pos);
		if (!replaced && line.find(prefix) == 0)
		{
			out += newLine;
			replaced = true;
		}
		else
		{
			out += line;
		}
		if (nl != std::string::npos)
			out += "\n";
		pos = (nl == std::string::npos) ? s.size() : (nl + 1);
	}
	return replaced && write_text_file(path, out);
}

static bool parse_kv_manifest(const std::string& path, std::map<std::string, std::string>& outKv)
{
	outKv.clear();
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	std::string line;
	bool firstLine = true;
	while (std::getline(in, line))
	{
		if (firstLine)
		{
			outKv["__magic__"] = line;
			firstLine = false;
			continue;
		}
		if (line.empty())
			continue;
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		outKv[line.substr(0, eq)] = line.substr(eq + 1);
	}
	return true;
}

static bool kv_get_u64(const std::map<std::string, std::string>& kv, const std::string& key, unsigned long long& out)
{
	std::map<std::string, std::string>::const_iterator it = kv.find(key);
	if (it == kv.end())
		return false;
	out = static_cast<unsigned long long>(strtoull(it->second.c_str(), NULL, 10));
	return true;
}

static bool kv_get_u32(const std::map<std::string, std::string>& kv, const std::string& key, unsigned int& out)
{
	unsigned long long v = 0ull;
	if (!kv_get_u64(kv, key, v))
		return false;
	out = static_cast<unsigned int>(v);
	return true;
}

static bool read_bytes_range(const std::string& path, unsigned long long offset, unsigned long long n, std::string& out)
{
	out.clear();
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
	if (!in)
		return false;
	std::string s;
	s.resize(static_cast<size_t>(n), '\0');
	if (n > 0ull)
		in.read(&s[0], static_cast<std::streamsize>(n));
	if (!in && n > 0ull)
		return false;
	out.swap(s);
	return true;
}

static bool flip_one_byte_in_file(const std::string& path, unsigned long long offset)
{
	std::fstream io(path.c_str(), std::ios::in | std::ios::out | std::ios::binary);
	if (!io)
		return false;
	io.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
	if (!io)
		return false;
	char c = 0;
	io.read(&c, 1);
	if (!io)
		return false;
	c = static_cast<char>(c ^ 0x5Au);
	io.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
	if (!io)
		return false;
	io.write(&c, 1);
	return static_cast<bool>(io);
}

struct CheckpointTensorLoc
{
	unsigned int shard;
	unsigned long long offsetBytes;
	unsigned long long bytes;
	CheckpointTensorLoc() : shard(0u), offsetBytes(0ull), bytes(0ull) {}
};

static bool checkpoint_find_tensor_index(const std::map<std::string, std::string>& kv,
                                         const std::string& tensorName,
                                         unsigned long long& outIdx)
{
	outIdx = 0ull;
	unsigned long long tensorCount = 0ull;
	if (!kv_get_u64(kv, "tensorCount", tensorCount))
		return false;
	for (unsigned long long i = 0ull; i < tensorCount; ++i)
	{
		std::ostringstream kn;
		kn << "tensor." << i << ".name";
		std::map<std::string, std::string>::const_iterator it = kv.find(kn.str());
		if (it == kv.end())
			return false;
		if (it->second == tensorName)
		{
			outIdx = i;
			return true;
		}
	}
	return false;
}

// In-memory token-id dataset for token language model checkpoint tests.
// (Copied from nn-test.cpp to avoid a shared UT dependency.)
class InMemoryTokenIdInput : public glades::DataInput
{
public:
	InMemoryTokenIdInput()
	    : padTokenId(-1),
	      scratchTok(0.0f),
	      scratchNext(0.0f),
	      one(1, 0.0f),
	      empty()
	{
	}

	void setTrainTokens(const std::vector<unsigned int>& toks, int pad)
	{
		padTokenId = pad;
		trainTok.clear();
		trainNextTok.clear();
		trainTok.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
			trainTok.push_back(static_cast<int>(toks[i]));
		build_next(trainTok, padTokenId, trainNextTok);
	}

	void mirrorTrainToTest()
	{
		testTok = trainTok;
		testNextTok = trainNextTok;
	}

	// DataInput API (no-op imports; dataset is constructed programmatically).
	virtual void import(shmea::GString, int = 0) {}
	virtual void import(const shmea::GTable&, int = 0) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int i) const
	{
		if (i >= trainTok.size())
			return empty;
		one[0] = static_cast<float>(trainTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{
		if (i >= trainNextTok.size())
			return empty;
		one[0] = static_cast<float>(trainNextTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{
		if (i >= testTok.size())
			return empty;
		one[0] = static_cast<float>(testTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{
		if (i >= testNextTok.size())
			return empty;
		one[0] = static_cast<float>(testNextTok[i]);
		return one;
	}

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainTok.size())
			return false;
		scratchTok = static_cast<float>(trainTok[index]);
		outData = &scratchTok;
		outSize = 1u;
		return true;
	}
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainNextTok.size())
			return false;
		scratchNext = static_cast<float>(trainNextTok[index]);
		outData = &scratchNext;
		outSize = 1u;
		return true;
	}
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testTok.size())
			return false;
		scratchTok = static_cast<float>(testTok[index]);
		outData = &scratchTok;
		outSize = 1u;
		return true;
	}
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testNextTok.size())
			return false;
		scratchNext = static_cast<float>(testNextTok[index]);
		outData = &scratchNext;
		outSize = 1u;
		return true;
	}

	// Token-id accessors (first-class ints).
	virtual bool getTrainTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainTok.size())
			return false;
		outTokenId = trainTok[index];
		return true;
	}
	virtual bool getTrainExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainNextTok.size())
			return false;
		outTokenId = trainNextTok[index];
		return true;
	}
	virtual bool getTestTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testTok.size())
			return false;
		outTokenId = testTok[index];
		return true;
	}
	virtual bool getTestExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testNextTok.size())
			return false;
		outTokenId = testNextTok[index];
		return true;
	}

	virtual unsigned int getTrainSize() const { return static_cast<unsigned int>(trainTok.size()); }
	virtual unsigned int getTestSize() const { return static_cast<unsigned int>(testTok.size()); }
	virtual unsigned int getFeatureCount() const { return 1u; }
	virtual int getType() const { return TEXT; }

private:
	static void build_next(const std::vector<int>& toks, int pad, std::vector<int>& outNext)
	{
		outNext.clear();
		outNext.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
		{
			if (i + 1u < toks.size())
				outNext.push_back(toks[i + 1u]);
			else
				outNext.push_back(pad);
		}
	}

	int padTokenId;
	std::vector<int> trainTok;
	std::vector<int> trainNextTok;
	std::vector<int> testTok;
	std::vector<int> testNextTok;

	mutable float scratchTok;
	mutable float scratchNext;
	mutable shmea::GVector<float> one;
	shmea::GVector<float> empty;
};

static std::string checkpoint_shard_path(const std::string& checkpointName, unsigned int shardIdx)
{
	std::ostringstream oss;
	oss << "database/checkpoints/" << checkpointName << "/shard_";
	oss.width(3);
	oss.fill('0');
	oss << shardIdx;
	oss << ".bin";
	return oss.str();
}

static bool checkpoint_find_tensor_loc(const std::map<std::string, std::string>& kv,
                                       const std::string& tensorName,
                                       CheckpointTensorLoc& outLoc)
{
	outLoc = CheckpointTensorLoc();
	unsigned long long tensorCount = 0ull;
	if (!kv_get_u64(kv, "tensorCount", tensorCount))
		return false;
	for (unsigned long long i = 0ull; i < tensorCount; ++i)
	{
		std::ostringstream kn;
		kn << "tensor." << i << ".name";
		std::map<std::string, std::string>::const_iterator it = kv.find(kn.str());
		if (it == kv.end())
			return false;
		if (it->second != tensorName)
			continue;

		std::ostringstream ks;
		std::ostringstream ko;
		std::ostringstream kb;
		ks << "tensor." << i << ".shard";
		ko << "tensor." << i << ".offsetBytes";
		kb << "tensor." << i << ".bytes";
		unsigned int shard = 0u;
		unsigned long long off = 0ull;
		unsigned long long bytes = 0ull;
		if (!kv_get_u32(kv, ks.str(), shard))
			return false;
		if (!kv_get_u64(kv, ko.str(), off))
			return false;
		if (!kv_get_u64(kv, kb.str(), bytes))
			return false;
		outLoc.shard = shard;
		outLoc.offsetBytes = off;
		outLoc.bytes = bytes;
		return true;
	}
	return false;
}

static bool read_u32_le(std::istream& in, unsigned int& outV)
{
	unsigned char b[4];
	in.read(reinterpret_cast<char*>(b), 4);
	if (!in)
		return false;
	outV = (static_cast<unsigned int>(b[0]) << 0) |
	       (static_cast<unsigned int>(b[1]) << 8) |
	       (static_cast<unsigned int>(b[2]) << 16) |
	       (static_cast<unsigned int>(b[3]) << 24);
	return true;
}

static bool weights_bin_header_ok(const std::string& path, unsigned int expectedNetType)
{
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	char magic[32];
	in.read(magic, 32);
	if (!in)
		return false;
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	if (wantLen > sizeof(magic) || std::memcmp(magic, want, wantLen) != 0)
		return false;
	unsigned int version = 0u;
	unsigned int netType = 0u;
	unsigned int r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, netType) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return false;
	if (version != 1u)
		return false;
	return netType == expectedNetType;
}
} // namespace

void NNSaveLoadUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("NN Save/Load Test (Unified model package)\n");
    printf("-----------------------------------\n");

    // Unit tests should be deterministic, but we still want a visible "the model ran"
    // signal in the output. The default callbacks also print a full graph dump on run end,
    // which is noisy for unit tests, so we use a minimal test console callback instead.
    class TestConsoleCallbacks : public glades::ITrainingCallbacks
    {
    public:
        virtual void onRunStart(const glades::NNetwork& net, int runType)
        {
            const glades::NNInfo* sk = net.getNNInfo();
            // IMPORTANT: NNInfo::getName() returns a GString by value. Calling .c_str() on that
            // temporary would yield a dangling pointer. Copy into a stable local first.
            const std::string name = (sk ? std::string(sk->getName().c_str()) : std::string("(unnamed)"));
            if (runType == glades::NNetwork::RUN_TRAIN)
                printf("[UT-NN] %s Training...\n", name.c_str());
            else if (runType == glades::NNetwork::RUN_TEST)
                printf("[UT-NN] %s Testing...\n", name.c_str());
        }

        virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
        {
            if (m.runType != glades::NNetwork::RUN_TRAIN)
                return false;

            if (m.outputType == glades::GMath::REGRESSION)
            {
                printf("[UT-NN] epoch=%d R2=%f%% MSE=%f MAE=%f RMSE=%f lr=%g(mult=%g) gradNorm=%g(scale=%g)\n",
                       m.epoch,
                       m.totalAccuracy,
                       m.totalError,
                       m.regMAE,
                       m.regRMSE,
                       m.learningRate,
                       m.lrMultiplier,
                       m.gradNorm,
                       m.gradNormScale);
            }
            else
            {
                printf("[UT-NN] epoch=%d acc=%f%% MCC=%f%% prec=%f%% recall=%f%% spec=%f%% f1=%f%%\n",
                       m.epoch,
                       m.classAccuracy,
                       m.classMCC,
                       m.classPrecision,
                       m.classRecall,
                       m.classSpecificity,
                       m.classF1);
            }

            return false;
        }

        virtual void onRunEnd(const glades::NNetwork&, int) {}
    };
    TestConsoleCallbacks testCb;

    // This test validates the production-facing persistence API:
    //   NNetwork::saveModel() / NNetwork::loadModel()
    //
    // It verifies:
    // - Package files are created (manifest + nninfo + weights)
    // - Weights round-trip correctly (tensor-first)
    // - netType, epochs, RNG seed, and TrainingConfig are restored from manifest

    // ============================
    // Case A: DFF round-trip
    // ============================
    {
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
        di->trainMatrix[0][0] = 1.0f;
        di->trainExpectedMatrix[0][0] = 0.0f;
        di->trainMatrix[1][0] = 2.0f;
        di->trainExpectedMatrix[1][0] = 0.0f;

        // Mirror train->test so the test split is well-formed.
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        // Build a minimal 1->1 regression NNInfo with deterministic graph weights.
        // Use a small non-zero LR so the run performs a real parameter update (visible in output),
        // while still being fully deterministic for this tiny dataset.
        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 2,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_dff", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
        net.setSeed(12345u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);

        // Set non-default TrainingConfig knobs to validate manifest persistence.
        glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
        cfg.minibatchSizeOverride = 7;
        cfg.tbpttWindowOverride = 13;
        cfg.globalGradClipNorm = 5.5f;
        cfg.perElementGradClip = 9.0f;
        cfg.lrSchedule.setStep(/*stepSize*/ 3, /*gamma*/ 0.25f);

        // Run a tiny train pass (1 epoch) so weights change deterministically.
        const glades::NNetworkStatus stTrain = net.train(di, &testCb);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_TrainStatus() Failed==============",
                 stTrain.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_EpochIncrement() Failed==============",
                 net.getEpochs() == 1);

        const std::string modelName = "ut_model_pkg_dff_v2";
        const glades::NNetworkStatus stSave = net.saveModel(modelName);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_SaveModel() Failed==============",
                 stSave.ok());

        // Basic package existence check (manifest + nninfo + weights).
        {
            std::ifstream mf(("database/models/" + modelName + "/manifest.txt").c_str());
            std::ifstream ni(("database/models/" + modelName + "/nninfo.csv").c_str());
            std::ifstream wt(("database/models/" + modelName + "/weights.bin").c_str(), std::ios::in | std::ios::binary);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_PackageFilesMissing() Failed==============",
                     static_cast<bool>(mf) && static_cast<bool>(ni) && static_cast<bool>(wt));
        }
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::DFF_WeightsHeaderBad() Failed==============",
		         weights_bin_header_ok("database/models/" + modelName + "/weights.bin", /*netType*/ 0u));

        // Verify manifest content includes our core metadata (magic/version/netType).
        {
            std::ifstream mf(("database/models/" + modelName + "/manifest.txt").c_str());
            std::string line;
            bool sawMagic = false;
            bool sawVersion = false;
            bool sawNetType = false;
            while (std::getline(mf, line))
            {
                if (line == "GLADES_MODEL")
                    sawMagic = true;
                if (line.find("version=") == 0)
                    sawVersion = true;
                if (line == "netType=0") // DFF
                    sawNetType = true;
            }
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_ManifestMissingFields() Failed==============",
                     sawMagic && sawVersion && sawNetType);
        }

        // Load into a fresh network and verify round-trip.
        glades::NNetwork net2(glades::NNetwork::TYPE_DFF);
        const glades::NNetworkStatus stLoad = net2.loadModel(modelName, di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_LoadModel() Failed==============",
                 stLoad.ok());

        // Verify metadata restored.
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_SeedRestored() Failed==============",
                 net2.getSeed() == 12345u);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_EpochsRestored() Failed==============",
                 net2.getEpochs() == 1);

        const glades::TrainingConfig& cfg2 = net2.getTrainingConfig();
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_TrainingConfigRestored() Failed==============",
                 cfg2.minibatchSizeOverride == cfg.minibatchSizeOverride &&
                 cfg2.tbpttWindowOverride == cfg.tbpttWindowOverride &&
                 cfg2.globalGradClipNorm == cfg.globalGradClipNorm &&
                 cfg2.perElementGradClip == cfg.perElementGradClip &&
                 cfg2.lrSchedule.type == cfg.lrSchedule.type &&
                 cfg2.lrSchedule.stepSizeEpochs == cfg.lrSchedule.stepSizeEpochs &&
                 cfg2.lrSchedule.gamma == cfg.lrSchedule.gamma);

        // Verify weights restored by re-saving and comparing weights.bin bytes verbatim.
        const std::string modelName2 = "ut_model_pkg_dff_v2_roundtrip";
        const glades::NNetworkStatus stSave2 = net2.saveModel(modelName2);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::DFF_SaveModelRoundTrip() Failed==============",
                 stSave2.ok());
        {
            std::string w1, w2;
            const bool ok1 = read_file_to_string("database/models/" + modelName + "/weights.bin", w1);
            const bool ok2 = read_file_to_string("database/models/" + modelName2 + "/weights.bin", w2);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_ReadWeightsFiles() Failed==============",
                     ok1 && ok2);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::DFF_WeightsRoundTrip() Failed==============",
                     w1 == w2);
        }

        delete di;
        delete info; // owns in/out
    }

    // ============================
    // Case B: GRU round-trip (weights only; no training)
    // ============================
    {
        printf("[UT-NN] GRU save/load round-trip (no training)\n");
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_gru", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_GRU);
        net.setSeed(777u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);
        // Initialize tensor weights once (no training).
        const glades::NNetworkStatus stInit = net.test(di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_InitTestStatus() Failed==============",
                 stInit.ok());

        const std::string modelName = "ut_model_pkg_gru_v2";
        const glades::NNetworkStatus stSave = net.saveModel(modelName);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_SaveModel() Failed==============",
                 stSave.ok());

        glades::NNetwork net2(glades::NNetwork::TYPE_GRU);
        const glades::NNetworkStatus stLoad = net2.loadModel(modelName, di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_LoadModel() Failed==============",
                 stLoad.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_SeedRestored() Failed==============",
                 net2.getSeed() == 777u);
        // Verify weights round-trip by re-saving and comparing weights.bin bytes verbatim.
        const std::string modelName2 = "ut_model_pkg_gru_v2_roundtrip";
        const glades::NNetworkStatus stSave2 = net2.saveModel(modelName2);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::GRU_SaveModelRoundTrip() Failed==============",
                 stSave2.ok());
        {
            std::string w1, w2;
            const bool ok1 = read_file_to_string("database/models/" + modelName + "/weights.bin", w1);
            const bool ok2 = read_file_to_string("database/models/" + modelName2 + "/weights.bin", w2);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::GRU_ReadWeightsFiles() Failed==============",
                     ok1 && ok2);
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::GRU_WeightsRoundTrip() Failed==============",
                     w1 == w2);
        }

        delete di;
        delete info; // owns in/hidden/out
    }

    // ============================
    // Case C: RNN round-trip (weights only; no training)
    // ============================
    {
        printf("[UT-NN] RNN save/load round-trip (no training)\n");
        glades::NumberInput* di = new glades::NumberInput();
        // Single short sequence length 3.
        di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        for (int t = 0; t < 3; ++t)
        {
            di->trainMatrix[t][0] = static_cast<float>(t);
            di->trainExpectedMatrix[t][0] = static_cast<float>(t);
        }
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
            /*size*/ 2,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_rnn", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_RNN);
        net.setSeed(888u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);
        const glades::NNetworkStatus stInit = net.test(di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::RNN_InitTestStatus() Failed==============",
                 stInit.ok());

        const std::string modelName = "ut_model_pkg_rnn_v2";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::RNN_SaveModel() Failed==============",
                 net.saveModel(modelName).ok());

        glades::NNetwork net2(glades::NNetwork::TYPE_RNN);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::RNN_LoadModel() Failed==============",
                 net2.loadModel(modelName, di).ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::RNN_SeedRestored() Failed==============",
                 net2.getSeed() == 888u);

        const std::string modelName2 = "ut_model_pkg_rnn_v2_roundtrip";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::RNN_SaveModelRoundTrip() Failed==============",
                 net2.saveModel(modelName2).ok());
        {
            std::string w1, w2;
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::RNN_ReadWeightsFiles() Failed==============",
                     read_file_to_string("database/models/" + modelName + "/weights.bin", w1) &&
                     read_file_to_string("database/models/" + modelName2 + "/weights.bin", w2));
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::RNN_WeightsRoundTrip() Failed==============",
                     w1 == w2);
        }

        delete di;
        delete info;
    }

    // ============================
    // Case D: LSTM round-trip (weights only; no training)
    // ============================
    {
        printf("[UT-NN] LSTM save/load round-trip (no training)\n");
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        for (int t = 0; t < 3; ++t)
        {
            di->trainMatrix[t][0] = static_cast<float>(t);
            di->trainExpectedMatrix[t][0] = static_cast<float>(t);
        }
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
            /*size*/ 2,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        ));
        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_lstm", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_LSTM);
        net.setSeed(999u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::LSTM_InitTestStatus() Failed==============",
                 net.test(di).ok());

        const std::string modelName = "ut_model_pkg_lstm_v2";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::LSTM_SaveModel() Failed==============",
                 net.saveModel(modelName).ok());

        glades::NNetwork net2(glades::NNetwork::TYPE_LSTM);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::LSTM_LoadModel() Failed==============",
                 net2.loadModel(modelName, di).ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::LSTM_SeedRestored() Failed==============",
                 net2.getSeed() == 999u);

        const std::string modelName2 = "ut_model_pkg_lstm_v2_roundtrip";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::LSTM_SaveModelRoundTrip() Failed==============",
                 net2.saveModel(modelName2).ok());
        {
            std::string w1, w2;
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::LSTM_ReadWeightsFiles() Failed==============",
                     read_file_to_string("database/models/" + modelName + "/weights.bin", w1) &&
                     read_file_to_string("database/models/" + modelName2 + "/weights.bin", w2));
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::LSTM_WeightsRoundTrip() Failed==============",
                     w1 == w2);
        }

        delete di;
        delete info;
    }

    // ============================
    // Case E: Transformer round-trip (weights only; no training)
    // ============================
    {
        printf("[UT-NN] Transformer save/load round-trip (no training)\n");
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
        for (int t = 0; t < 3; ++t)
        {
            di->trainMatrix[t][0] = static_cast<float>(t);
            di->trainExpectedMatrix[t][0] = static_cast<float>(t);
        }
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
            /*size*/ 8,                 // dModel
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f));

        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_save_load_transformer", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
        net.setSeed(2026u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
		}
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::TR_InitTestStatus() Failed==============",
                 net.test(di).ok());

        const std::string modelName = "ut_model_pkg_tr_v2";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::TR_SaveModel() Failed==============",
                 net.saveModel(modelName).ok());

		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TR_WeightsHeaderBad() Failed==============",
		         weights_bin_header_ok("database/models/" + modelName + "/weights.bin", /*netType*/ 4u));

        glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::TR_LoadModel() Failed==============",
                 net2.loadModel(modelName, di).ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::TR_SeedRestored() Failed==============",
                 net2.getSeed() == 2026u);

        const std::string modelName2 = "ut_model_pkg_tr_v2_roundtrip";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::TR_SaveModelRoundTrip() Failed==============",
                 net2.saveModel(modelName2).ok());
        {
            std::string w1, w2;
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::TR_ReadWeightsFiles() Failed==============",
                     read_file_to_string("database/models/" + modelName + "/weights.bin", w1) &&
                     read_file_to_string("database/models/" + modelName2 + "/weights.bin", w2));
            G_assert(__FILE__, __LINE__,
                     "==============NNSaveLoad::TR_WeightsRoundTrip() Failed==============",
                     w1 == w2);
        }

        delete di;
        delete info;
    }

    // ============================
    // Case F: Manifest netType corruption + override rescue
    // ============================
    {
        printf("[UT-NN] Manifest netType mismatch + loadModel override\n");
        glades::NumberInput* di = new glades::NumberInput();
        di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di->testMatrix = di->trainMatrix;
        di->testExpectedMatrix = di->trainExpectedMatrix;

        auto in = shmea::make_gpointer<glades::InputLayerInfo>(
            /*batchSize*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
            /*activationParam*/ 1.0f
        );
        std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
        auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
        glades::NNInfo* info = new glades::NNInfo("ut_manifest_override", in, hidden, out);

        glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
        net.setSeed(111u);
        net.getTerminatorMutable().setEpoch(1);
        net.getTerminatorMutable().setAccuracy(0);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::Override_InitTestStatus() Failed==============",
                 net.test(di).ok());
        const std::string modelName = "ut_model_pkg_override";
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::Override_SaveModel() Failed==============",
                 net.saveModel(modelName).ok());

        // Corrupt manifest netType to mismatch weights netType (weights remain TYPE_DFF).
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::Override_CorruptManifest() Failed==============",
                 replace_line_prefix_in_file("database/models/" + modelName + "/manifest.txt", "netType=", "netType=1"));

        glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
        const glades::NNetworkStatus stBad = netBad.loadModel(modelName, di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::Override_LoadShouldFail() Failed==============",
                 !stBad.ok());

        // Override netType to match weights file, bypassing manifest netType corruption.
        glades::NNetwork netOk(glades::NNetwork::TYPE_DFF);
        const glades::NNetworkStatus stOk = netOk.loadModel(modelName, di, /*netTypeOverride*/ glades::NNetwork::TYPE_DFF);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::Override_LoadWithOverride() Failed==============",
                 stOk.ok());

        delete di;
        delete info;
    }

    // ============================
    // Case G: Negative - missing manifest should fail (modern-only)
    // ============================
    {
        printf("[UT-NN] loadModel fails without manifest.txt\n");
        // Ensure directory exists but no manifest.txt.
        mkdir_if_missing("database");
        mkdir_if_missing("database/models");
        mkdir_if_missing("database/models/ut_missing_manifest_pkg");

        glades::NumberInput di;
        di.trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di.trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
        di.testMatrix = di.trainMatrix;
        di.testExpectedMatrix = di.trainExpectedMatrix;

        glades::NNetwork net(glades::NNetwork::TYPE_DFF);
        const glades::NNetworkStatus st = net.loadModel("ut_missing_manifest_pkg", &di);
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::MissingManifest_ShouldFail() Failed==============",
                 !st.ok());
        G_assert(__FILE__, __LINE__,
                 "==============NNSaveLoad::MissingManifest_Message() Failed==============",
                 st.message.find("manifest.txt") != std::string::npos);
    }

	// ============================
	// Case H: Checkpoint round-trip (DFF) with optimizer state
	// ============================
	{
		printf("[UT-NN] Checkpoint save/load (DFF + momentum optimizer state)\n");

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		for (int i = 0; i < 4; ++i)
		{
			di->trainMatrix[i][0] = static_cast<float>(i);
			di->trainMatrix[i][1] = static_cast<float>(i * 2);
			di->trainExpectedMatrix[i][0] = static_cast<float>(i % 2);
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.9f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::TANH,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    /*size*/ 4,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.9f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::TANH,
		    /*activationParam*/ 1.0f
		));
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_ckpt_dff", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(4242u);
		net.getTerminatorMutable().setEpoch(2);
		net.getTerminatorMutable().setAccuracy(0);

		// Force some real optimizer state (momentum) to be created.
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_TrainStatus() Failed==============",
		         net.train(di, &testCb).ok());

		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.maxShardBytes = 4096u; // small to exercise sharding
		ccfg.includeOptimizerState = true;

		const std::string ckptName = "ut_checkpoint_dff_opt";
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_SaveCheckpoint() Failed==============",
		         net.saveCheckpoint(ckptName, ccfg).ok());

		// Verify checkpoint manifest includes optimizer tensors (at least one vW).
		{
			std::map<std::string, std::string> kv;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_ReadManifest() Failed==============",
			         parse_kv_manifest("database/checkpoints/" + ckptName + "/manifest.txt", kv));
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_Magic() Failed==============",
			         kv["__magic__"] == "GLADES_CHECKPOINT");
			// Checkpoint metadata: explicit encoding markers.
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_Version() Failed==============",
			         kv.find("version") != kv.end() && kv["version"] == "1");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_FileEndian() Failed==============",
			         kv.find("file.endian") != kv.end() && kv["file.endian"] == "little");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_TensorEncoding() Failed==============",
			         kv.find("file.tensorEncoding") != kv.end() && kv["file.tensorEncoding"] == "raw_le");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_IncludeOpt() Failed==============",
			         kv["includeOptimizerState"] == "1");

			// Per-tensor metadata: check a known tensor has dtype/shape markers.
			{
				unsigned long long tensorCount = 0ull;
				G_assert(__FILE__, __LINE__,
				         "==============NNSaveLoad::CKPT_DFF_TensorCountPresent() Failed==============",
				         kv_get_u64(kv, "tensorCount", tensorCount));
				bool found = false;
				for (unsigned long long i = 0ull; i < tensorCount; ++i)
				{
					std::ostringstream kn; kn << "tensor." << i << ".name";
					if (kv.find(kn.str()) == kv.end())
						continue;
					if (kv.find(kn.str())->second != "dff.t0.W")
						continue;
					found = true;
					std::ostringstream kd; kd << "tensor." << i << ".dtype";
					std::ostringstream ksh; ksh << "tensor." << i << ".shape";
					std::ostringstream ke; ke << "tensor." << i << ".elemBytes";
					G_assert(__FILE__, __LINE__,
					         "==============NNSaveLoad::CKPT_DFF_TensorDType() Failed==============",
					         kv.find(kd.str()) != kv.end() && kv[kd.str()] == "f32");
					G_assert(__FILE__, __LINE__,
					         "==============NNSaveLoad::CKPT_DFF_TensorElemBytes() Failed==============",
					         kv.find(ke.str()) != kv.end() && kv[ke.str()] == "4");
					// Shape is [out,in] for dff.t0.W => hidden(4) x input(2).
					G_assert(__FILE__, __LINE__,
					         "==============NNSaveLoad::CKPT_DFF_TensorShape() Failed==============",
					         kv.find(ksh.str()) != kv.end() && kv[ksh.str()] == "4,2");
					break;
				}
				G_assert(__FILE__, __LINE__,
				         "==============NNSaveLoad::CKPT_DFF_TensorMetaFound() Failed==============",
				         found);
			}

			// Nuanced check: the momentum tensor exists AND contains non-zero bytes after training.
			CheckpointTensorLoc vLoc;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_FindMomentumLoc() Failed==============",
			         checkpoint_find_tensor_loc(kv, "dff.t0.vW", vLoc));
			std::string blob;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_ReadMomentumBlob() Failed==============",
			         read_bytes_range(checkpoint_shard_path(ckptName, vLoc.shard), vLoc.offsetBytes, vLoc.bytes, blob));
			bool anyNonZero = false;
			for (size_t i = 0; i < blob.size(); ++i)
			{
				if (blob[i] != 0)
				{
					anyNonZero = true;
					break;
				}
			}
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_MomentumNonZero() Failed==============",
			         anyNonZero);
		}
		{
			std::string manifest;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_ReadManifestText() Failed==============",
			         read_file_to_string("database/checkpoints/" + ckptName + "/manifest.txt", manifest));
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_ManifestMentionsMomentum() Failed==============",
			         manifest.find("dff.t0.vW") != std::string::npos);
		}

		// Load into a fresh net and re-save checkpoint; shard bytes should be identical.
		glades::NNetwork net2(glades::NNetwork::TYPE_DFF);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_LoadCheckpoint() Failed==============",
		         net2.loadCheckpoint(ckptName, di).ok());
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_SeedRestored() Failed==============",
		         net2.getSeed() == 4242u);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_EpochsRestored() Failed==============",
		         net2.getEpochs() == net.getEpochs());

		const std::string ckptName2 = "ut_checkpoint_dff_opt_roundtrip";
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_DFF_SaveCheckpointRoundTrip() Failed==============",
		         net2.saveCheckpoint(ckptName2, ccfg).ok());
		{
			// Compare shard_000.bin only; if sharding differs, manifest parsing below will catch it.
			std::string s1, s2;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_ReadShard0() Failed==============",
			         read_file_to_string("database/checkpoints/" + ckptName + "/shard_000.bin", s1) &&
			         read_file_to_string("database/checkpoints/" + ckptName2 + "/shard_000.bin", s2));
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_DFF_Shard0RoundTrip() Failed==============",
			         s1 == s2);
		}

		delete di;
		delete info;
	}

	// ============================
	// Case I: Checkpoint sharding + corruption detection + override rescue
	// ============================
	{
		printf("[UT-NN] Checkpoint sharding + corruption detection + netType override rescue\n");

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(3, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(2, 0.0f));
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    /*size*/ 64, // large enough that multiple tensors will span shards with small shard size
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		));
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(2, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_ckpt_shard", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(9001u);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		// Initialize tensors.
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Shard_InitTestStatus() Failed==============",
		         net.test(di).ok());

		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.maxShardBytes = 2048u; // aggressive sharding
		ccfg.includeOptimizerState = true;

		const std::string ckptName = "ut_checkpoint_sharded";
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Shard_SaveCheckpoint() Failed==============",
		         net.saveCheckpoint(ckptName, ccfg).ok());

		std::map<std::string, std::string> kv;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Shard_ParseManifest() Failed==============",
		         parse_kv_manifest("database/checkpoints/" + ckptName + "/manifest.txt", kv));
		unsigned long long shardCount = 0ull;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Shard_ShardCountPresent() Failed==============",
		         kv_get_u64(kv, "shardCount", shardCount));
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Shard_HasMultipleShards() Failed==============",
		         shardCount >= 2ull);

		// Corrupt a shard byte inside a known tensor region and ensure checksum failure.
		{
			CheckpointTensorLoc loc;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Shard_FindTensorLoc() Failed==============",
			         checkpoint_find_tensor_loc(kv, "dff.t0.W", loc));
			// Flip one byte somewhere inside the tensor blob.
			const std::string shardPath = "database/checkpoints/" + ckptName + "/shard_";
			std::ostringstream sp;
			sp << "database/checkpoints/" << ckptName << "/shard_";
			sp.width(3); sp.fill('0'); sp << loc.shard;
			sp << ".bin";
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Shard_FlipByte() Failed==============",
			         flip_one_byte_in_file(sp.str(), loc.offsetBytes + (loc.bytes / 2ull)));

			glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
			const glades::NNetworkStatus st = netBad.loadCheckpoint(ckptName, di);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Shard_CorruptionShouldFail() Failed==============",
			         !st.ok());
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Shard_CorruptionMessage() Failed==============",
			         st.message.find("checksum") != std::string::npos || st.message.find("checksum/read") != std::string::npos);
		}

		// netType override rescue: corrupt netType in manifest and verify override can still load.
		{
			// Restore by re-saving a fresh checkpoint to avoid using the already-corrupted shard data.
			const std::string ckpt2 = "ut_checkpoint_override";
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Override_SaveCheckpoint() Failed==============",
			         net.saveCheckpoint(ckpt2, ccfg).ok());
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Override_CorruptNetType() Failed==============",
			         replace_line_prefix_in_file("database/checkpoints/" + ckpt2 + "/manifest.txt", "netType=", "netType=1"));

			glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Override_LoadShouldFail() Failed==============",
			         !netBad.loadCheckpoint(ckpt2, di).ok());

			glades::NNetwork netOk(glades::NNetwork::TYPE_DFF);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Override_LoadWithOverride() Failed==============",
			         netOk.loadCheckpoint(ckpt2, di, glades::NNetwork::TYPE_DFF).ok());
		}

		delete di;
		delete info;
	}

	// ============================
	// Case J: Negative - saveCheckpoint without initialized tensors should fail
	// ============================
	{
		printf("[UT-NN] saveCheckpoint fails if tensors are not initialized\n");
		glades::NumberInput di;
		di.trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di.trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di.testMatrix = di.trainMatrix;
		di.testExpectedMatrix = di.trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_ckpt_precond", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		// Intentionally do NOT call train/test to initialize tensors.
		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.includeOptimizerState = true;
		const glades::NNetworkStatus st = net.saveCheckpoint("ut_checkpoint_precond_fail", ccfg);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Precond_SaveShouldFail() Failed==============",
		         !st.ok());
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Precond_Message() Failed==============",
		         st.message.find("initialized") != std::string::npos || st.message.find("tensors") != std::string::npos);

		delete info;
	}

	// ============================
	// Case K: Tokenizer artifacts round-trip (saveModel/loadModel)
	// ============================
	{
		printf("[UT-NN] Tokenizer artifacts save/load round-trip\n");

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_tokenizer_artifacts", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(13579u);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);

		// Initialize tensors once so saveModel has weights to persist.
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_InitTestStatus() Failed==============",
		         net.test(di).ok());

		glades::NNetwork::TokenizerArtifacts ta;
		ta.type = "custom";
		ta.vocab.push_back("<pad>");       // id 0
		ta.vocab.push_back("<bos>");       // id 1
		ta.vocab.push_back("<eos>");       // id 2
		ta.vocab.push_back("hello");       // id 3
		ta.vocab.push_back("hello world"); // id 4 (contains space)
		ta.padTokenId = 0;
		ta.bosTokenId = 1;
		ta.eosTokenId = 2;
		ta.unkTokenId = -1;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_Set() Failed==============",
		         net.setTokenizerArtifacts(ta).ok());

		const std::string modelName = "ut_model_pkg_with_tokenizer";
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_SaveModel() Failed==============",
		         net.saveModel(modelName).ok());

		// Verify tokenizer artifact files exist.
		{
			std::ifstream tm(("database/models/" + modelName + "/tokenizer/manifest.txt").c_str());
			std::ifstream vb(("database/models/" + modelName + "/tokenizer/vocab.bin").c_str(), std::ios::in | std::ios::binary);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TokArtifacts_FilesMissing() Failed==============",
			         static_cast<bool>(tm) && static_cast<bool>(vb));
		}

		glades::NNetwork net2(glades::NNetwork::TYPE_DFF);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_LoadModel() Failed==============",
		         net2.loadModel(modelName, di).ok());
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_Present() Failed==============",
		         net2.hasTokenizerArtifacts());

		const glades::NNetwork::TokenizerArtifacts& tb = net2.getTokenizerArtifacts();
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_TypeRoundTrip() Failed==============",
		         tb.type == ta.type);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_VocabSizeRoundTrip() Failed==============",
		         tb.vocab.size() == ta.vocab.size());
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_SpecialIdsRoundTrip() Failed==============",
		         tb.padTokenId == ta.padTokenId &&
		         tb.bosTokenId == ta.bosTokenId &&
		         tb.eosTokenId == ta.eosTokenId &&
		         tb.unkTokenId == ta.unkTokenId);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokArtifacts_TokenContentRoundTrip() Failed==============",
		         tb.vocab.size() >= 5u && tb.vocab[4] == "hello world");

		delete di;
		delete info;
	}

	// ============================
	// Case L: Tokenizer vocab corruption should be detected (checksum)
	// ============================
	{
		printf("[UT-NN] Tokenizer vocab corruption detection (checksum)\n");

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_tokenizer_corrupt", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_InitTestStatus() Failed==============",
		         net.test(di).ok());

		glades::NNetwork::TokenizerArtifacts ta;
		ta.type = "custom";
		ta.vocab.push_back("<pad>"); // ensures bytes exist after header
		ta.vocab.push_back("<bos>");
		ta.vocab.push_back("<eos>");
		ta.vocab.push_back("x");
		ta.padTokenId = 0;
		ta.bosTokenId = 1;
		ta.eosTokenId = 2;
		ta.unkTokenId = -1;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_Set() Failed==============",
		         net.setTokenizerArtifacts(ta).ok());

		const std::string modelName = "ut_model_pkg_tok_corrupt";
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_SaveModel() Failed==============",
		         net.saveModel(modelName).ok());

		const std::string vocabPath = "database/models/" + modelName + "/tokenizer/vocab.bin";
		// Flip a byte inside the first token's bytes (offset: 4-byte count + 4-byte len = 8).
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_FlipByte() Failed==============",
		         flip_one_byte_in_file(vocabPath, 8ull));

		glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
		const glades::NNetworkStatus st = netBad.loadModel(modelName, di);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_LoadShouldFail() Failed==============",
		         !st.ok());
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::TokCorrupt_MessageMentionsChecksum() Failed==============",
		         st.message.find("checksum") != std::string::npos || st.message.find("fnv1a64") != std::string::npos);

		delete di;
		delete info;
	}

	// ============================
	// Case N: Checkpoint manifest metadata strictness (format v1-only)
	// ============================
	{
		printf("[UT-NN] Checkpoint manifest strict metadata validation\n");

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f
		);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    /*size*/ 4,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::TANH,
		    /*activationParam*/ 1.0f
		));
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_ckpt_manifest_strict", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_InitTestStatus() Failed==============",
		         net.test(di).ok());

		const std::string ckpt = "ut_checkpoint_manifest_strict";
		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.includeOptimizerState = false;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_SaveCheckpoint() Failed==============",
		         net.saveCheckpoint(ckpt, ccfg).ok());

		const std::string manifestPath = "database/checkpoints/" + ckpt + "/manifest.txt";
		std::map<std::string, std::string> kv;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_ReadManifest() Failed==============",
		         parse_kv_manifest(manifestPath, kv));

		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_Version() Failed==============",
		         kv["version"] == "1");
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_FileEndian() Failed==============",
		         kv["file.endian"] == "little");
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_TensorEncoding() Failed==============",
		         kv["file.tensorEncoding"] == "raw_le");

		unsigned long long wi = 0ull;
		G_assert(__FILE__, __LINE__,
		         "==============NNSaveLoad::CKPT_Strict_FindW() Failed==============",
		         checkpoint_find_tensor_index(kv, "dff.t0.W", wi));
		{
			std::ostringstream kd, ke, kr, ksh, kl;
			kd << "tensor." << wi << ".dtype";
			ke << "tensor." << wi << ".elemBytes";
			kr << "tensor." << wi << ".rank";
			ksh << "tensor." << wi << ".shape";
			kl << "tensor." << wi << ".layout";
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_W_DType() Failed==============",
			         kv[kd.str()] == "f32");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_W_ElemBytes() Failed==============",
			         kv[ke.str()] == "4");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_W_Rank() Failed==============",
			         kv[kr.str()] == "2");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_W_Shape() Failed==============",
			         kv[ksh.str()] == "4,1");
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_W_Layout() Failed==============",
			         kv[kl.str()] == "row-major");
		}

		// Negative: file.endian must be "little".
		{
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_CorruptEndian() Failed==============",
			         replace_line_prefix_in_file(manifestPath, "file.endian=", "file.endian=big"));
			glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
			const glades::NNetworkStatus st = netBad.loadCheckpoint(ckpt, di);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_LoadShouldFailEndian() Failed==============",
			         !st.ok());
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_MessageMentionsEndian() Failed==============",
			         st.message.find("endian") != std::string::npos);
		}

		// Restore endian, then negative: per-tensor dtype mismatch should fail.
		{
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_RestoreEndian() Failed==============",
			         replace_line_prefix_in_file(manifestPath, "file.endian=", "file.endian=little"));

			std::ostringstream kdline;
			kdline << "tensor." << wi << ".dtype=";
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_CorruptDType() Failed==============",
			         replace_line_prefix_in_file(manifestPath, kdline.str(), kdline.str() + "bf16"));
			glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
			const glades::NNetworkStatus st = netBad.loadCheckpoint(ckpt, di);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_LoadShouldFailDType() Failed==============",
			         !st.ok());
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_MessageMentionsDType() Failed==============",
			         st.message.find("dtype") != std::string::npos);
		}

		// Restore dtype, then negative: shape mismatch (even if element count matches) should fail.
		{
			std::ostringstream kdline, kshline;
			kdline << "tensor." << wi << ".dtype=";
			kshline << "tensor." << wi << ".shape=";
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_RestoreDType() Failed==============",
			         replace_line_prefix_in_file(manifestPath, kdline.str(), kdline.str() + "f32"));
			// Swap shape dims from [4,1] -> [1,4] (same count).
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_CorruptShape() Failed==============",
			         replace_line_prefix_in_file(manifestPath, kshline.str(), kshline.str() + "1,4"));
			glades::NNetwork netBad(glades::NNetwork::TYPE_DFF);
			const glades::NNetworkStatus st = netBad.loadCheckpoint(ckpt, di);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_LoadShouldFailShape() Failed==============",
			         !st.ok());
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::CKPT_Strict_MessageMentionsShape() Failed==============",
			         st.message.find("shape") != std::string::npos || st.message.find("rank") != std::string::npos);
		}

		delete di;
		delete info;
	}

	// ============================
	// Case O: Transformer checkpoints - tokenModel gating + token-LM metadata
	// ============================
	{
		printf("[UT-NN] Transformer checkpoint tokenModel gating + token-LM tensors\n");

		// --- Regression transformer: must NOT persist token-LM tensors.
		{
			glades::NumberInput di;
			di.trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
			di.trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
			di.testMatrix = di.trainMatrix;
			di.testExpectedMatrix = di.trainExpectedMatrix;

			auto in = shmea::make_gpointer<glades::InputLayerInfo>(
			    /*batchSize*/ 1,
			    /*learningRate*/ 0.0f,
			    /*momentumFactor*/ 0.0f,
			    /*weightDecay1*/ 0.0f,
			    /*weightDecay2*/ 0.0f,
			    /*pDropout*/ 0.0f,
			    /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f
			);
			std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
			hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
			    /*size*/ 16, // dModel
			    /*learningRate*/ 0.0f,
			    /*momentumFactor*/ 0.0f,
			    /*weightDecay1*/ 0.0f,
			    /*weightDecay2*/ 0.0f,
			    /*pDropout*/ 0.0f,
			    /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f
			));
			auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_tr_ckpt_reg", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
				cfg.transformer.nHeadsOverride = 2;
				cfg.transformer.dFFOverride = 32;
				cfg.transformer.enableTokenEmbedding = false; // regression mode
				cfg.transformer.tieEmbeddings = false;
				cfg.transformer.vocabSizeOverride = 0;
				cfg.transformer.padTokenId = -1;
			}
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_InitTestStatus() Failed==============",
			         net.test(&di).ok());

			const std::string ckpt = "ut_checkpoint_tr_regression";
			glades::NNetwork::CheckpointConfig ccfg;
			ccfg.includeOptimizerState = true; // ADAMW requires this to load
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_SaveCheckpoint() Failed==============",
			         net.saveCheckpoint(ckpt, ccfg).ok());

			std::string manifest;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_ReadManifestText() Failed==============",
			         read_file_to_string("database/checkpoints/" + ckpt + "/manifest.txt", manifest));

			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_NoTokE() Failed==============",
			         manifest.find("tr.tokE") == std::string::npos);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_NoLmBias() Failed==============",
			         manifest.find("tr.lmBias") == std::string::npos);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_NoVTokE() Failed==============",
			         manifest.find("tr.vTokE") == std::string::npos);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_NoMLmBias() Failed==============",
			         manifest.find("tr.mLmBias") == std::string::npos);

			glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrReg_LoadCheckpoint() Failed==============",
			         net2.loadCheckpoint(ckpt, &di).ok());

			delete info;
		}

		// --- Token LM transformer: must persist token-LM tensors with correct shapes.
		{
			InMemoryTokenIdInput di;
			const unsigned int vocab = 16u;
			const int pad = static_cast<int>(vocab - 1u);
			std::vector<unsigned int> toks;
			toks.push_back(1u);
			toks.push_back(2u);
			toks.push_back(3u);
			toks.push_back(4u);
			di.setTrainTokens(toks, pad);
			di.mirrorTrainToTest();

			auto in = shmea::make_gpointer<glades::InputLayerInfo>(
			    /*batchSize*/ 1,
			    /*learningRate*/ 0.0f,
			    /*momentumFactor*/ 0.0f,
			    /*weightDecay1*/ 0.0f,
			    /*weightDecay2*/ 0.0f,
			    /*pDropout*/ 0.0f,
			    /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f
			);
			std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
			hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
			    /*size*/ 16, // dModel
			    /*learningRate*/ 0.0f,
			    /*momentumFactor*/ 0.0f,
			    /*weightDecay1*/ 0.0f,
			    /*weightDecay2*/ 0.0f,
			    /*pDropout*/ 0.0f,
			    /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f
			));
			auto out = shmea::make_gpointer<glades::OutputLayerInfo>(vocab, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_tr_ckpt_tokenlm", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
				cfg.transformer.nHeadsOverride = 2;
				cfg.transformer.dFFOverride = 32;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
				cfg.transformer.padTokenId = pad;
			}
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_InitTestStatus() Failed==============",
			         net.test(&di).ok());

			const std::string ckpt = "ut_checkpoint_tr_tokenlm";
			glades::NNetwork::CheckpointConfig ccfg;
			ccfg.includeOptimizerState = true;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_SaveCheckpoint() Failed==============",
			         net.saveCheckpoint(ckpt, ccfg).ok());

			const std::string manifestPath = "database/checkpoints/" + ckpt + "/manifest.txt";
			std::map<std::string, std::string> kv;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_ReadManifestKV() Failed==============",
			         parse_kv_manifest(manifestPath, kv));

			unsigned long long ei = 0ull;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_FindTokE() Failed==============",
			         checkpoint_find_tensor_index(kv, "tr.tokE", ei));
			{
				std::ostringstream ksh;
				ksh << "tensor." << ei << ".shape";
				G_assert(__FILE__, __LINE__,
				         "==============NNSaveLoad::TrTokLM_TokE_Shape() Failed==============",
				         kv[ksh.str()] == "16,16");
			}

			unsigned long long bi = 0ull;
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_FindLmBias() Failed==============",
			         checkpoint_find_tensor_index(kv, "tr.lmBias", bi));
			{
				std::ostringstream ksh;
				ksh << "tensor." << bi << ".shape";
				G_assert(__FILE__, __LINE__,
				         "==============NNSaveLoad::TrTokLM_LmBias_Shape() Failed==============",
				         kv[ksh.str()] == "16");
			}

			glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TrTokLM_LoadCheckpoint() Failed==============",
			         net2.loadCheckpoint(ckpt, &di).ok());

			delete info;
		}
	}

	// ============================
	// Case M: TokenizerArtifacts API validation (duplicates / range / empty)
	// ============================
	{
		printf("[UT-NN] TokenizerArtifacts validation\n");

		glades::NNetwork net(glades::NNetwork::TYPE_DFF);

		// Empty vocab should fail.
		{
			glades::NNetwork::TokenizerArtifacts ta;
			ta.type = "custom";
			const glades::NNetworkStatus st = net.setTokenizerArtifacts(ta);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TokValidate_EmptyVocabShouldFail() Failed==============",
			         !st.ok());
		}

		// Duplicate tokens should fail.
		{
			glades::NNetwork::TokenizerArtifacts ta;
			ta.type = "custom";
			ta.vocab.push_back("a");
			ta.vocab.push_back("a");
			const glades::NNetworkStatus st = net.setTokenizerArtifacts(ta);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TokValidate_DuplicateShouldFail() Failed==============",
			         !st.ok());
		}

		// Out-of-range special id should fail.
		{
			glades::NNetwork::TokenizerArtifacts ta;
			ta.type = "custom";
			ta.vocab.push_back("a");
			ta.vocab.push_back("b");
			ta.padTokenId = 999;
			const glades::NNetworkStatus st = net.setTokenizerArtifacts(ta);
			G_assert(__FILE__, __LINE__,
			         "==============NNSaveLoad::TokValidate_SpecialIdRangeShouldFail() Failed==============",
			         !st.ok());
		}
	}

    printf("\n============================================================\n");
}
