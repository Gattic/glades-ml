// MappedNumberInput implementation.
#include "MappedNumberInput.h"

#include <sys/stat.h>

namespace {

static inline std::string to_std_string(const shmea::GString& s)
{
	return std::string(s.c_str());
}

static inline bool path_is_dir(const std::string& p)
{
	struct stat st;
	if (stat(p.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode);
}

static inline bool stat_mtime(const std::string& p, time_t& outMTime)
{
	struct stat st;
	if (stat(p.c_str(), &st) != 0)
		return false;
	outMTime = st.st_mtime;
	return true;
}

} // namespace

// Link anchor: see note in MappedMatrix.cpp.
extern "C" void glades_link_anchor_mappednumberinput()
{
	// no-op
}

glades::MappedNumberInput::MappedNumberInput()
    : DataInput(),
      trainX(),
      trainY(),
      testX(),
      testY(),
      loaded(false),
      featureCountCached(0u),
      expectedCountCached(0u),
      scratchRow(),
      scratchY(),
      emptyRow(),
      lastErr(),
      lastImportStatus(glades::NNetworkStatus::OK, std::string())
{
	// Ensure no categorical artifacts (this is pure numeric tensor input).
	trainingOHEMaps.clear();
	testingOHEMaps.clear();
	trainingFeatureIsCategorical.clear();
	testingFeatureIsCategorical.clear();
}

glades::MappedNumberInput::~MappedNumberInput()
{
	clear();
}

void glades::MappedNumberInput::clear()
{
	trainX.close();
	trainY.close();
	testX.close();
	testY.close();

	loaded = false;
	featureCountCached = 0u;
	expectedCountCached = 0u;
	lastErr.clear();
	lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());

	scratchRow.clear();
	scratchY.clear();
	emptyRow.clear();

	trainingOHEMaps.clear();
	testingOHEMaps.clear();
	trainingFeatureIsCategorical.clear();
	testingFeatureIsCategorical.clear();
}

bool glades::MappedNumberInput::openDir(const std::string& dirPath, std::string* errMsg)
{
	if (!path_is_dir(dirPath))
	{
		if (errMsg) *errMsg = "MappedNumberInput::import: path is not a directory";
		return false;
	}

	std::string err;
	const std::string trX = dirPath + "/train.x.gcol";
	const std::string trY = dirPath + "/train.y.gcol";
	const std::string teX = dirPath + "/test.x.gcol";
	const std::string teY = dirPath + "/test.y.gcol";

	if (!trainX.openReadOnly(trX, &err))
	{
		if (errMsg) *errMsg = std::string("failed to open train.x.gcol: ") + err;
		return false;
	}
	if (!trainY.openReadOnly(trY, &err))
	{
		if (errMsg) *errMsg = std::string("failed to open train.y.gcol: ") + err;
		return false;
	}

	// Determine "freshness" of train files. If test files exist but are older than train,
	// treat them as stale and ignore them.
	//
	// Rationale:
	// - Unit tests and some workflows reuse an on-disk dataset directory and may overwrite
	//   train.* without removing previously-existing test.*.
	// - In such cases, it's safer to interpret the dataset as "train-only" unless the test
	//   split appears to have been produced alongside (or after) the current train split.
	time_t trXTime = 0, trYTime = 0;
	(void)stat_mtime(trX, trXTime);
	(void)stat_mtime(trY, trYTime);
	const time_t trainLatest = (trXTime > trYTime) ? trXTime : trYTime;

	// Optional test split.
	// If either test file is missing/invalid, treat as "no test split".
	std::string errTeX;
	std::string errTeY;
	bool allowTest = true;
	time_t teXTime = 0, teYTime = 0;
	if (!stat_mtime(teX, teXTime) || !stat_mtime(teY, teYTime))
		allowTest = false;
	// If trainLatest==0 (stat failed), fall back to allowing test if files exist.
	if (allowTest && trainLatest != 0 && (teXTime < trainLatest || teYTime < trainLatest))
		allowTest = false;

	const bool okTeX = allowTest ? testX.openReadOnly(teX, &errTeX) : false;
	const bool okTeY = allowTest ? testY.openReadOnly(teY, &errTeY) : false;
	if (!(okTeX && okTeY))
	{
		testX.close();
		testY.close();
	}

	// Basic shape invariants.
	if (trainX.rows() != trainY.rows())
	{
		if (errMsg) *errMsg = "train.x rows != train.y rows";
		return false;
	}
	if (testX.isOpen() && testY.isOpen())
	{
		if (testX.rows() != testY.rows())
		{
			if (errMsg) *errMsg = "test.x rows != test.y rows";
			return false;
		}
		if (testX.cols() != trainX.cols())
		{
			if (errMsg) *errMsg = "test.x cols != train.x cols";
			return false;
		}
		if (testY.cols() != trainY.cols())
		{
			if (errMsg) *errMsg = "test.y cols != train.y cols";
			return false;
		}
	}

	featureCountCached = static_cast<unsigned int>(trainX.cols());
	expectedCountCached = static_cast<unsigned int>(trainY.cols());
	if (featureCountCached == 0u || expectedCountCached == 0u)
	{
		if (errMsg) *errMsg = "invalid feature/output dims (0)";
		return false;
	}

	// Set min/max to neutral values (unknown without scanning).
	min = 0.0f;
	max = 0.0f;

	return true;
}

void glades::MappedNumberInput::import(shmea::GString path, int /*standardizeFlag*/)
{
	clear();

	const std::string p = to_std_string(path);
	if (p.empty())
	{
		lastErr = "MappedNumberInput::import: empty path";
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, lastErr);
		loaded = false;
		return;
	}
	std::string err;
	if (!openDir(p, &err))
	{
		lastErr = err;
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, err);
		loaded = false;
		return;
	}

	loaded = true;
	lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

void glades::MappedNumberInput::import(const shmea::GTable& /*rawTable*/, int /*standardizeFlag*/)
{
	// Not supported: this input is intended for on-disk memory-mapped matrices.
	clear();
	lastErr = "MappedNumberInput::import(GTable): not supported (use directory import with .gcol files)";
	lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, lastErr);
	loaded = false;
}

unsigned int glades::MappedNumberInput::getTrainSize() const
{
	return loaded ? static_cast<unsigned int>(trainX.rows()) : 0u;
}

unsigned int glades::MappedNumberInput::getTestSize() const
{
	if (!loaded)
		return 0u;
	return testX.isOpen() ? static_cast<unsigned int>(testX.rows()) : 0u;
}

unsigned int glades::MappedNumberInput::getFeatureCount() const
{
	return loaded ? featureCountCached : 0u;
}

bool glades::MappedNumberInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!loaded)
		return false;
	if (static_cast<unsigned long long>(index) >= trainX.rows())
		return false;

	const float* p = trainX.rowPtr(static_cast<unsigned long long>(index));
	if (!p)
		return false;
	outData = p;
	outSize = featureCountCached;
	return true;
}

bool glades::MappedNumberInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!loaded)
		return false;
	if (static_cast<unsigned long long>(index) >= trainY.rows())
		return false;

	const float* p = trainY.rowPtr(static_cast<unsigned long long>(index));
	if (!p)
		return false;
	outData = p;
	outSize = expectedCountCached;
	return true;
}

bool glades::MappedNumberInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!loaded || !testX.isOpen())
		return false;
	if (static_cast<unsigned long long>(index) >= testX.rows())
		return false;

	const float* p = testX.rowPtr(static_cast<unsigned long long>(index));
	if (!p)
		return false;
	outData = p;
	outSize = featureCountCached;
	return true;
}

bool glades::MappedNumberInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!loaded || !testY.isOpen())
		return false;
	if (static_cast<unsigned long long>(index) >= testY.rows())
		return false;

	const float* p = testY.rowPtr(static_cast<unsigned long long>(index));
	if (!p)
		return false;
	outData = p;
	outSize = expectedCountCached;
	return true;
}

shmea::GVector<float> glades::MappedNumberInput::getTrainRow(unsigned int index) const
{
	const float* p = NULL;
	unsigned int n = 0u;
	if (!getTrainRowView(index, p, n) || !p || n == 0u)
		return emptyRow;
	scratchRow = shmea::GVector<float>(n, 0.0f);
	for (unsigned int i = 0; i < n; ++i)
		scratchRow[i] = p[i];
	return scratchRow;
}

shmea::GVector<float> glades::MappedNumberInput::getTrainExpectedRow(unsigned int index) const
{
	const float* p = NULL;
	unsigned int n = 0u;
	if (!getTrainExpectedRowView(index, p, n) || !p || n == 0u)
		return emptyRow;
	scratchY = shmea::GVector<float>(n, 0.0f);
	for (unsigned int i = 0; i < n; ++i)
		scratchY[i] = p[i];
	return scratchY;
}

shmea::GVector<float> glades::MappedNumberInput::getTestRow(unsigned int index) const
{
	const float* p = NULL;
	unsigned int n = 0u;
	if (!getTestRowView(index, p, n) || !p || n == 0u)
		return shmea::GVector<float>();
	scratchRow = shmea::GVector<float>(n, 0.0f);
	for (unsigned int i = 0; i < n; ++i)
		scratchRow[i] = p[i];
	return scratchRow;
}

shmea::GVector<float> glades::MappedNumberInput::getTestExpectedRow(unsigned int index) const
{
	const float* p = NULL;
	unsigned int n = 0u;
	if (!getTestExpectedRowView(index, p, n) || !p || n == 0u)
		return shmea::GVector<float>();
	scratchY = shmea::GVector<float>(n, 0.0f);
	for (unsigned int i = 0; i < n; ++i)
		scratchY[i] = p[i];
	return scratchY;
}

int glades::MappedNumberInput::getType() const
{
	// Still represents tabular numeric input.
	return CSV;
}

