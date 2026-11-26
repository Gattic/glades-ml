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
#include "QuaternionInput.h"
#include "../rng.h"

#include <cmath>

namespace {

static glades::Quaternion sample_unit_quaternion(glades::rng::Engine& eng)
{
	for (unsigned int tries = 0u; tries < 64u; ++tries)
	{
		glades::Quaternion q(
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng));
		const float n = q.norm();
		if (n > 1e-6f)
		{
			q /= n;
			if (q.r < 0.0f)
				q = -q;
			return q;
		}
	}
	return glades::Quaternion::one();
}

static glades::Quaternion sample_unit_vec3_pure(glades::rng::Engine& eng)
{
	for (unsigned int tries = 0u; tries < 64u; ++tries)
	{
		const float x = glades::rng::standard_normal(eng);
		const float y = glades::rng::standard_normal(eng);
		const float z = glades::rng::standard_normal(eng);
		const float n = sqrtf((x * x) + (y * y) + (z * z));
		if (n > 1e-6f)
			return glades::Quaternion(0.0f, x / n, y / n, z / n);
	}
	return glades::Quaternion(0.0f, 1.0f, 0.0f, 0.0f);
}

static glades::Quaternion sample_axis_angle_quaternion(glades::rng::Engine& eng,
                                                       float minAngleRad,
                                                       float maxAngleRad)
{
	const glades::Quaternion axis = sample_unit_vec3_pure(eng);
	const float theta = glades::rng::uniform_float(eng, minAngleRad, maxAngleRad);
	const float half = 0.5f * theta;
	const float c = cosf(half);
	const float s = sinf(half);
	glades::Quaternion q(c, axis.i * s, axis.j * s, axis.k * s);
	if (q.r < 0.0f)
		q = -q;
	return q;
}

static glades::Quaternion sample_hemisphere_vec3(glades::rng::Engine& eng,
                                                 bool upperHemisphere,
                                                 float minAbsZ)
{
	for (unsigned int tries = 0u; tries < 256u; ++tries)
	{
		glades::Quaternion dir = sample_unit_vec3_pure(eng);
		if (upperHemisphere && dir.k > minAbsZ)
			return dir;
		if (!upperHemisphere && dir.k < -minAbsZ)
			return dir;
	}
	return upperHemisphere
	    ? glades::Quaternion(0.0f, 0.0f, 0.0f, 1.0f)
	    : glades::Quaternion(0.0f, 0.0f, 0.0f, -1.0f);
}

static glades::Quaternion sample_signed_k_unit_quaternion(glades::rng::Engine& eng,
                                                          bool positiveK,
                                                          float minAbsK)
{
	for (unsigned int tries = 0u; tries < 256u; ++tries)
	{
		glades::Quaternion q = sample_unit_quaternion(eng);
		if (positiveK && q.k > minAbsK)
			return q;
		if (!positiveK && q.k < -minAbsK)
			return q;
	}
	return positiveK
	    ? glades::Quaternion(0.70710678f, 0.0f, 0.0f, 0.70710678f)
	    : glades::Quaternion(0.70710678f, 0.0f, 0.0f, -0.70710678f);
}

static glades::Quaternion quat_rot_z(float angleRad)
{
	const float half = 0.5f * angleRad;
	return glades::Quaternion(cosf(half), 0.0f, 0.0f, sinf(half));
}

static glades::Quaternion quat_rot_x(float angleRad)
{
	const float half = 0.5f * angleRad;
	return glades::Quaternion(cosf(half), sinf(half), 0.0f, 0.0f);
}

static glades::Quaternion sample_inclination_rotation(glades::rng::Engine& eng,
                                                      float minIncRad,
                                                      float maxIncRad)
{
	static const float kTwoPi = 6.28318530717958647692f;
	const float omegaBig = glades::rng::uniform_float(eng, 0.0f, kTwoPi);
	const float omegaSmall = glades::rng::uniform_float(eng, 0.0f, kTwoPi);
	const float cosHi = cosf(minIncRad);
	const float cosLo = cosf(maxIncRad);
	const float cosInc = glades::rng::uniform_float(eng, cosLo, cosHi);
	const float clamped = (cosInc > 1.0f) ? 1.0f : ((cosInc < -1.0f) ? -1.0f : cosInc);
	const float inc = acosf(clamped);
	glades::Quaternion q = glades::Quaternion::mul(
	    quat_rot_z(omegaBig),
	    glades::Quaternion::mul(quat_rot_x(inc), quat_rot_z(omegaSmall)));
	q = q.normalized();
	if (q.r < 0.0f)
		q = -q;
	return q;
}

static float pure_vec_dot(const glades::Quaternion& a, const glades::Quaternion& b)
{
	return (a.i * b.i) + (a.j * b.j) + (a.k * b.k);
}

static glades::Quaternion pure_vec_cross(const glades::Quaternion& a, const glades::Quaternion& b)
{
	return glades::Quaternion(
	    0.0f,
	    (a.j * b.k) - (a.k * b.j),
	    (a.k * b.i) - (a.i * b.k),
	    (a.i * b.j) - (a.j * b.i));
}

static void compute_h2o_pes_and_force(const glades::Quaternion& r1,
                                      const glades::Quaternion& r2,
                                      float& outEnergy,
                                      glades::Quaternion& outForceO)
{
	// Triatomic H2O-like anharmonic Morse stretch + harmonic cosine-angle bend:
	//   E = De * [(1 - exp(-alpha*(d1 - r0)))^2 + (1 - exp(-alpha*(d2 - r0)))^2]
	//       + kTheta * (cos(theta) - cos(theta0))^2
	// Central Oxygen atom force: F_O = -dE/dx_O = dE/dr1 + dE/dr2.
	static const double kR0 = 0.9584;
	static const double kCosTheta0 = -0.2495354328; // cos(104.45 deg)
	static const double kDe = 1.0;
	static const double kAlpha = 1.6;
	static const double kTheta = 0.9;

	const double r1x = static_cast<double>(r1.i);
	const double r1y = static_cast<double>(r1.j);
	const double r1z = static_cast<double>(r1.k);
	const double r2x = static_cast<double>(r2.i);
	const double r2y = static_cast<double>(r2.j);
	const double r2z = static_cast<double>(r2.k);

	const double d1 = sqrt((r1x * r1x) + (r1y * r1y) + (r1z * r1z));
	const double d2 = sqrt((r2x * r2x) + (r2y * r2y) + (r2z * r2z));
	const double invD1 = (d1 > 1e-12) ? (1.0 / d1) : 0.0;
	const double invD2 = (d2 > 1e-12) ? (1.0 / d2) : 0.0;

	const double u1x = r1x * invD1, u1y = r1y * invD1, u1z = r1z * invD1;
	const double u2x = r2x * invD2, u2y = r2y * invD2, u2z = r2z * invD2;

	const double cosT = (u1x * u2x) + (u1y * u2y) + (u1z * u2z);
	const double dc = cosT - kCosTheta0;

	const double e1 = exp(-kAlpha * (d1 - kR0));
	const double e2 = exp(-kAlpha * (d2 - kR0));
	const double m1 = 1.0 - e1;
	const double m2 = 1.0 - e2;

	const double energy = kDe * ((m1 * m1) + (m2 * m2)) + kTheta * (dc * dc);

	const double rad1 = 2.0 * kDe * kAlpha * m1 * e1;
	const double rad2 = 2.0 * kDe * kAlpha * m2 * e2;
	const double angCoeff = 2.0 * kTheta * dc;

	const double dEdr1x = rad1 * u1x + angCoeff * (u2x - cosT * u1x) * invD1;
	const double dEdr1y = rad1 * u1y + angCoeff * (u2y - cosT * u1y) * invD1;
	const double dEdr1z = rad1 * u1z + angCoeff * (u2z - cosT * u1z) * invD1;

	const double dEdr2x = rad2 * u2x + angCoeff * (u1x - cosT * u2x) * invD2;
	const double dEdr2y = rad2 * u2y + angCoeff * (u1y - cosT * u2y) * invD2;
	const double dEdr2z = rad2 * u2z + angCoeff * (u1z - cosT * u2z) * invD2;

	outEnergy = static_cast<float>(energy);
	outForceO = glades::Quaternion(
	    0.0f,
	    static_cast<float>(dEdr1x + dEdr2x),
	    static_cast<float>(dEdr1y + dEdr2y),
	    static_cast<float>(dEdr1z + dEdr2z));
}

} // namespace

namespace glades {

QuaternionInput::QuaternionInput()
    : NumberInput(),
      inputPackingMode(QUAT_PACK_GROUPS_OF_4),
      outputReadoutMode(QUAT_READOUT_REAL_PART),
      quatInputCount(0u),
      quatOutputCount(0u),
      realOutputCount(0u),
      lastStatus()
{
}

QuaternionInput::~QuaternionInput()
{
	clearQuaternionBuffers();
}

void QuaternionInput::clearQuaternionBuffers()
{
	quatInputCount = 0u;
	quatOutputCount = 0u;
	realOutputCount = 0u;
	trainQuatInputs.clear();
	trainQuatTargets.clear();
	testQuatInputs.clear();
	testQuatTargets.clear();
	classLabels.clear();
}

bool QuaternionInput::rebuildQuaternionBuffers(QuaternionInputPackingMode packing,
                                               QuaternionOutputReadoutMode readout)
{
	inputPackingMode = packing;
	outputReadoutMode = readout;
	trainQuatInputs.clear();
	trainQuatTargets.clear();
	testQuatInputs.clear();
	testQuatTargets.clear();
	quatInputCount = 0u;
	quatOutputCount = 0u;
	realOutputCount = 0u;

	const unsigned int nTrain = getTrainSize();
	const unsigned int nTest = getTestSize();
	const unsigned int realIn = getFeatureCount();
	unsigned int realOut = getFixedTrainExpectedRowSize();
	if (realOut == 0u && nTest > 0u)
	{
		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (getTestExpectedRowView(0u, yPtr, ySz))
			realOut = ySz;
	}

	if (realIn == 0u || realOut == 0u || (nTrain == 0u && nTest == 0u))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::EMPTY_DATA,
		                            "QuaternionInput::rebuildQuaternionBuffers: empty dataset or zero feature/output width");
		return false;
	}

	const unsigned int qIn = quaternionInputCount(realIn, packing);
	const unsigned int qOut = quaternionOutputCount(realOut, readout);
	if (qIn == 0u || qOut == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::rebuildQuaternionBuffers: invalid quaternion dimensions");
		return false;
	}

	quatInputCount = qIn;
	quatOutputCount = qOut;
	realOutputCount = realOut;

	trainQuatInputs.assign(static_cast<size_t>(nTrain) * static_cast<size_t>(qIn), Quaternion::zero());
	trainQuatTargets.assign(static_cast<size_t>(nTrain) * static_cast<size_t>(qOut), Quaternion::zero());
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const float* xPtr = NULL;
		unsigned int xSz = 0u;
		if (!getTrainRowView(r, xPtr, xSz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read train input row");
			return false;
		}
		Quaternion* dstIn = &trainQuatInputs[static_cast<size_t>(r) * static_cast<size_t>(qIn)];
		encodeRealToQuaternions(xPtr, xSz, realIn, packing, dstIn, qIn);
		for (unsigned int q = 0u; q < qIn; ++q)
		{
			if (!dstIn[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite train feature value");
				return false;
			}
		}

		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (!getTrainExpectedRowView(r, yPtr, ySz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read train expected row");
			return false;
		}
		Quaternion* dstOut = &trainQuatTargets[static_cast<size_t>(r) * static_cast<size_t>(qOut)];
		encodeTargetsToQuaternions(yPtr, ySz, realOut, readout, dstOut, qOut);
		for (unsigned int q = 0u; q < qOut; ++q)
		{
			if (!dstOut[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite train target value");
				return false;
			}
		}
	}

	testQuatInputs.assign(static_cast<size_t>(nTest) * static_cast<size_t>(qIn), Quaternion::zero());
	testQuatTargets.assign(static_cast<size_t>(nTest) * static_cast<size_t>(qOut), Quaternion::zero());
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const float* xPtr = NULL;
		unsigned int xSz = 0u;
		if (!getTestRowView(r, xPtr, xSz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read test input row");
			return false;
		}
		Quaternion* dstIn = &testQuatInputs[static_cast<size_t>(r) * static_cast<size_t>(qIn)];
		encodeRealToQuaternions(xPtr, xSz, realIn, packing, dstIn, qIn);
		for (unsigned int q = 0u; q < qIn; ++q)
		{
			if (!dstIn[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite test feature value");
				return false;
			}
		}

		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (!getTestExpectedRowView(r, yPtr, ySz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read test expected row");
			return false;
		}
		Quaternion* dstOut = &testQuatTargets[static_cast<size_t>(r) * static_cast<size_t>(qOut)];
		encodeTargetsToQuaternions(yPtr, ySz, realOut, readout, dstOut, qOut);
		for (unsigned int q = 0u; q < qOut; ++q)
		{
			if (!dstOut[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite test target value");
				return false;
			}
		}
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadTabularWithSplit(const shmea::GTable& rawTable,
                                           const TrainTestSplitConfig& splitCfg,
                                           int standardizeFlag,
                                           QuaternionInputPackingMode packing,
                                           QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (!importWithSplit(rawTable, splitCfg, standardizeFlag))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadTabularWithSplit: NumberInput::importWithSplit failed");
		return false;
	}
	return rebuildQuaternionBuffers(packing, readout);
}

bool QuaternionInput::loadIris(const std::string& filePath,
                               const TrainTestSplitConfig& splitCfg,
                               int standardizeFlag,
                               QuaternionInputPackingMode packing,
                               QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() == 0u || tbl.numberOfCols() != 5u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: expected 5-column Iris CSV (4 features + 1 class)");
		return false;
	}
	tbl.toggleOutput(4u);

	if (!loadTabularWithSplit(tbl, splitCfg, standardizeFlag, packing, readout))
		return false;

	if (getFeatureCount() != 4u || getRealOutputCount() != 3u)
	{
		clearQuaternionBuffers();
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: expected 4 numeric features and 3 encoded classes");
		return false;
	}

	classLabels.clear();
	if (trainingOHEMaps.size() > 4u && trainingOHEMaps[4u])
	{
		const unsigned int nClasses = trainingOHEMaps[4u]->size();
		classLabels.reserve(nClasses);
		for (unsigned int c = 0u; c < nClasses; ++c)
			classLabels.push_back(std::string(trainingOHEMaps[4u]->classAt(c).c_str()));
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadXOR(const std::string& filePath,
                              QuaternionInputPackingMode packing,
                              QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadXOR: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() != 4u || tbl.numberOfCols() != 3u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadXOR: expected 4x3 XOR truth table CSV");
		return false;
	}

	trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
	for (unsigned int r = 0u; r < 4u; ++r)
	{
		trainMatrix[r][0] = tbl.getCell(r, 0u).getFloat();
		trainMatrix[r][1] = tbl.getCell(r, 1u).getFloat();
		trainExpectedMatrix[r][0] = tbl.getCell(r, 2u).getFloat();
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = 2u;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadParityFile(const std::string& filePath,
                                     QuaternionInputPackingMode packing,
                                     QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadParityFile: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	const unsigned int nRows = tbl.numberOfRows();
	const unsigned int nCols = tbl.numberOfCols();
	if (nRows == 0u || nCols < 2u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadParityFile: expected >= 1 row and >= 2 columns");
		return false;
	}

	const unsigned int nIn = nCols - 1u;
	trainMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(nIn, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nRows; ++r)
	{
		for (unsigned int c = 0u; c < nIn; ++c)
			trainMatrix[r][c] = tbl.getCell(r, c).getFloat();
		trainExpectedMatrix[r][0] = tbl.getCell(r, nIn).getFloat();
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = nIn;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadBinaryParity(unsigned int nBits,
                                       QuaternionInputPackingMode packing,
                                       QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nBits == 0u || nBits > 8u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadBinaryParity: nBits must be in [1, 8]");
		return false;
	}

	const unsigned int nRows = (1u << nBits);
	trainMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(nBits, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int mask = 0u; mask < nRows; ++mask)
	{
		unsigned int parity = 0u;
		for (unsigned int b = 0u; b < nBits; ++b)
		{
			const unsigned int bit = (mask >> (nBits - 1u - b)) & 1u;
			trainMatrix[mask][b] = static_cast<float>(bit);
			parity ^= bit;
		}
		trainExpectedMatrix[mask][0] = static_cast<float>(parity);
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = nBits;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadWine(const std::string& filePath,
                               const TrainTestSplitConfig& splitCfg,
                               int standardizeFlag,
                               QuaternionInputPackingMode packing,
                               QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() == 0u || tbl.numberOfCols() != 14u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: expected 14-column Wine CSV (13 features + 1 class)");
		return false;
	}
	tbl.toggleOutput(13u);

	if (!loadTabularWithSplit(tbl, splitCfg, standardizeFlag, packing, readout))
		return false;

	if (getFeatureCount() != 13u || getRealOutputCount() != 3u)
	{
		clearQuaternionBuffers();
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: expected 13 numeric features and 3 encoded classes");
		return false;
	}

	classLabels.clear();
	if (trainingOHEMaps.size() > 13u && trainingOHEMaps[13u])
	{
		const unsigned int nClasses = trainingOHEMaps[13u]->size();
		classLabels.reserve(nClasses);
		for (unsigned int c = 0u; c < nClasses; ++c)
			classLabels.push_back(std::string(trainingOHEMaps[13u]->classAt(c).c_str()));
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadDowJonesOHLC(const std::string& trainPath,
                                       const std::string& testPath,
                                       float* outTargetMean,
                                       float* outTargetStd,
                                       QuaternionInputPackingMode packing,
                                       QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (trainPath.empty() || testPath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadDowJonesOHLC: empty train/test path");
		return false;
	}

	shmea::GTable trainTbl(shmea::GString(trainPath.c_str()), ',', shmea::GTable::TYPE_FILE);
	shmea::GTable testTbl(shmea::GString(testPath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (trainTbl.numberOfRows() == 0u || testTbl.numberOfRows() == 0u ||
	    trainTbl.numberOfCols() < 13u || testTbl.numberOfCols() < 13u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadDowJonesOHLC: invalid Dow Jones CSV dimensions");
		return false;
	}

	// Columns in dow_jones_index_{train,test}.csv:
	//   3=open, 4=high, 5=low, 6=close, 12=next_weeks_close.
	const unsigned int ohlcCols[4] = { 3u, 4u, 5u, 6u };
	const unsigned int targetCol = 12u;
	const unsigned int nTrain = trainTbl.numberOfRows();
	const unsigned int nTest = testTbl.numberOfRows();

	double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
	double var[4] = { 0.0, 0.0, 0.0, 0.0 };
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
			mean[c] += static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat());
	}
	for (unsigned int c = 0u; c < 4u; ++c)
		mean[c] /= static_cast<double>(nTrain);

	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double d = static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat()) - mean[c];
			var[c] += d * d;
		}
	}

	double stdv[4] = { 1.0, 1.0, 1.0, 1.0 };
	for (unsigned int c = 0u; c < 4u; ++c)
	{
		const double s = sqrt(var[c] / static_cast<double>(nTrain));
		stdv[c] = (s > 1e-8) ? s : 1.0;
	}

	const double tMean = mean[3];
	const double tStd = stdv[3];
	if (outTargetMean)
		*outTargetMean = static_cast<float>(tMean);
	if (outTargetStd)
		*outTargetStd = static_cast<float>(tStd);

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double v = static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat());
			trainMatrix[r][c] = static_cast<float>((v - mean[c]) / stdv[c]);
		}
		const double y = static_cast<double>(trainTbl.getCell(r, targetCol).getFloat());
		trainExpectedMatrix[r][0] = static_cast<float>((y - tMean) / tStd);
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double v = static_cast<double>(testTbl.getCell(r, ohlcCols[c]).getFloat());
			testMatrix[r][c] = static_cast<float>((v - mean[c]) / stdv[c]);
		}
		const double y = static_cast<double>(testTbl.getCell(r, targetCol).getFloat());
		testExpectedMatrix[r][0] = static_cast<float>((y - tMean) / tStd);
	}

	featureCountCached = 4u;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadSO3Rotation(unsigned int nTrain,
                                      unsigned int nTest,
                                      unsigned int seed,
                                      bool oodHemisphereAndAngle,
                                      QuaternionInputPackingMode packing,
                                      QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadSO3Rotation: nTrain and nTest must be > 0");
		return false;
	}

	static const float kPi = 3.14159265358979323846f;
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const float radius = glades::rng::uniform_float(eng, 0.5f, 1.5f);
		const Quaternion vDir = oodHemisphereAndAngle
		    ? sample_hemisphere_vec3(eng, true, 0.25f)
		    : sample_unit_vec3_pure(eng);
		const Quaternion v(0.0f, vDir.i * radius, vDir.j * radius, vDir.k * radius);
		const Quaternion qRot = oodHemisphereAndAngle
		    ? sample_axis_angle_quaternion(eng, 0.0f, 0.5f * kPi)
		    : sample_unit_quaternion(eng);
		const Quaternion vRot = Quaternion::rotateVec(qRot, v);

		trainMatrix[r][0] = 0.0f;
		trainMatrix[r][1] = v.i;
		trainMatrix[r][2] = v.j;
		trainMatrix[r][3] = v.k;
		trainMatrix[r][4] = qRot.r;
		trainMatrix[r][5] = qRot.i;
		trainMatrix[r][6] = qRot.j;
		trainMatrix[r][7] = qRot.k;

		trainExpectedMatrix[r][0] = 0.0f;
		trainExpectedMatrix[r][1] = vRot.i;
		trainExpectedMatrix[r][2] = vRot.j;
		trainExpectedMatrix[r][3] = vRot.k;
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const float radius = glades::rng::uniform_float(eng, 0.5f, 1.5f);
		const Quaternion vDir = oodHemisphereAndAngle
		    ? sample_hemisphere_vec3(eng, false, 0.25f)
		    : sample_unit_vec3_pure(eng);
		const Quaternion v(0.0f, vDir.i * radius, vDir.j * radius, vDir.k * radius);
		const Quaternion qRot = oodHemisphereAndAngle
		    ? sample_axis_angle_quaternion(eng, 0.5f * kPi, kPi)
		    : sample_unit_quaternion(eng);
		const Quaternion vRot = Quaternion::rotateVec(qRot, v);

		testMatrix[r][0] = 0.0f;
		testMatrix[r][1] = v.i;
		testMatrix[r][2] = v.j;
		testMatrix[r][3] = v.k;
		testMatrix[r][4] = qRot.r;
		testMatrix[r][5] = qRot.i;
		testMatrix[r][6] = qRot.j;
		testMatrix[r][7] = qRot.k;

		testExpectedMatrix[r][0] = 0.0f;
		testExpectedMatrix[r][1] = vRot.i;
		testExpectedMatrix[r][2] = vRot.j;
		testExpectedMatrix[r][3] = vRot.k;
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadSO3Composition(unsigned int nTrain,
                                         unsigned int nTest,
                                         unsigned int seed,
                                         bool oodRotatedFrame,
                                         QuaternionInputPackingMode packing,
                                         QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadSO3Composition: nTrain and nTest must be > 0");
		return false;
	}

	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const Quaternion q1 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, true, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion q2 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, true, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion qOut = Quaternion::mul(q1, q2);

		trainMatrix[r][0] = q1.r;
		trainMatrix[r][1] = q1.i;
		trainMatrix[r][2] = q1.j;
		trainMatrix[r][3] = q1.k;
		trainMatrix[r][4] = q2.r;
		trainMatrix[r][5] = q2.i;
		trainMatrix[r][6] = q2.j;
		trainMatrix[r][7] = q2.k;

		trainExpectedMatrix[r][0] = qOut.r;
		trainExpectedMatrix[r][1] = qOut.i;
		trainExpectedMatrix[r][2] = qOut.j;
		trainExpectedMatrix[r][3] = qOut.k;
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const Quaternion q1 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, false, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion q2 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, false, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion qOut = Quaternion::mul(q1, q2);

		testMatrix[r][0] = q1.r;
		testMatrix[r][1] = q1.i;
		testMatrix[r][2] = q1.j;
		testMatrix[r][3] = q1.k;
		testMatrix[r][4] = q2.r;
		testMatrix[r][5] = q2.i;
		testMatrix[r][6] = q2.j;
		testMatrix[r][7] = q2.k;

		testExpectedMatrix[r][0] = qOut.r;
		testExpectedMatrix[r][1] = qOut.i;
		testExpectedMatrix[r][2] = qOut.j;
		testExpectedMatrix[r][3] = qOut.k;
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadMolecularPES(unsigned int nTrain,
                                       unsigned int nTest,
                                       unsigned int seed,
                                       bool oodOrientation,
                                       QuaternionInputPackingMode packing,
                                       QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadMolecularPES: nTrain and nTest must be > 0");
		return false;
	}

	static const float kDegToRad = 3.14159265358979323846f / 180.0f;
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));

	for (unsigned int split = 0u; split < 2u; ++split)
	{
		const bool isTrain = (split == 0u);
		const unsigned int nRows = isTrain ? nTrain : nTest;
		shmea::GMatrix& xMat = isTrain ? trainMatrix : testMatrix;
		shmea::GMatrix& yMat = isTrain ? trainExpectedMatrix : testExpectedMatrix;

		for (unsigned int r = 0u; r < nRows; ++r)
		{
			const float d1 = glades::rng::uniform_float(eng, 0.76f, 1.24f);
			const float d2 = glades::rng::uniform_float(eng, 0.76f, 1.24f);
			const float theta = glades::rng::uniform_float(eng, 75.0f * kDegToRad, 135.0f * kDegToRad);
			const float halfT = 0.5f * theta;

			const Quaternion r1Loc(0.0f, d1 * cosf(halfT), d1 * sinf(halfT), 0.0f);
			const Quaternion r2Loc(0.0f, d2 * cosf(halfT), -d2 * sinf(halfT), 0.0f);

			const Quaternion uRot = oodOrientation
			    ? (isTrain ? sample_axis_angle_quaternion(eng, 0.0f, 30.0f * kDegToRad)
			               : sample_axis_angle_quaternion(eng, 60.0f * kDegToRad, 180.0f * kDegToRad))
			    : sample_unit_quaternion(eng);

			const Quaternion r1 = Quaternion::rotateVec(uRot, r1Loc);
			const Quaternion r2 = Quaternion::rotateVec(uRot, r2Loc);

			float energy = 0.0f;
			Quaternion forceO;
			compute_h2o_pes_and_force(r1, r2, energy, forceO);

			xMat[r][0] = 0.0f;
			xMat[r][1] = r1.i;
			xMat[r][2] = r1.j;
			xMat[r][3] = r1.k;
			xMat[r][4] = 0.0f;
			xMat[r][5] = r2.i;
			xMat[r][6] = r2.j;
			xMat[r][7] = r2.k;

			yMat[r][0] = energy;
			yMat[r][1] = forceO.i;
			yMat[r][2] = forceO.j;
			yMat[r][3] = forceO.k;
		}
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadMolecularChirality(unsigned int nTrain,
                                             unsigned int nTest,
                                             unsigned int seed,
                                             bool oodOrientation,
                                             QuaternionInputPackingMode packing,
                                             QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadMolecularChirality: nTrain and nTest must be > 0");
		return false;
	}

	static const float kDegToRad = 3.14159265358979323846f / 180.0f;
	static const float kLigandCharge[3] = { 0.25f, 0.50f, 0.75f };
	static const float kBaseAzimuthDeg[3] = { 0.0f, 120.0f, 240.0f };

	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(12u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(3u, 0.0f));
	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(12u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(3u, 0.0f));

	for (unsigned int split = 0u; split < 2u; ++split)
	{
		const bool isTrain = (split == 0u);
		const unsigned int nRows = isTrain ? nTrain : nTest;
		shmea::GMatrix& xMat = isTrain ? trainMatrix : testMatrix;
		shmea::GMatrix& yMat = isTrain ? trainExpectedMatrix : testExpectedMatrix;

		// Build balanced class schedule and Fisher-Yates shuffle deterministically.
		std::vector<unsigned int> clsOrder(nRows, 0u);
		for (unsigned int r = 0u; r < nRows; ++r)
			clsOrder[r] = r % 3u;
		for (unsigned int r = nRows; r > 1u; --r)
		{
			const unsigned int j = glades::rng::uniform_uint(eng, 0u, r - 1u);
			const unsigned int tmp = clsOrder[r - 1u];
			clsOrder[r - 1u] = clsOrder[j];
			clsOrder[j] = tmp;
		}

		for (unsigned int r = 0u; r < nRows; ++r)
		{
			const unsigned int cls = clsOrder[r];
			float hApex = 0.0f;
			if (cls == 0u)
				hApex = glades::rng::uniform_float(eng, -0.04f, 0.04f);
			else if (cls == 1u)
				hApex = glades::rng::uniform_float(eng, 0.28f, 0.60f);
			else
				hApex = -glades::rng::uniform_float(eng, 0.28f, 0.60f);

			const Quaternion uRot = oodOrientation
			    ? (isTrain ? sample_axis_angle_quaternion(eng, 0.0f, 35.0f * kDegToRad)
			               : sample_axis_angle_quaternion(eng, 60.0f * kDegToRad, 180.0f * kDegToRad))
			    : sample_unit_quaternion(eng);

			for (unsigned int k = 0u; k < 3u; ++k)
			{
				const float az = (kBaseAzimuthDeg[k] + glades::rng::uniform_float(eng, -14.0f, 14.0f)) * kDegToRad;
				const float radial = glades::rng::uniform_float(eng, 0.85f, 1.15f);
				const float zNoise = glades::rng::uniform_float(eng, -0.025f, 0.025f);
				const Quaternion bLoc(0.0f, radial * cosf(az), radial * sinf(az), hApex + zNoise);
				const Quaternion bRot = Quaternion::rotateVec(uRot, bLoc);

				const unsigned int off = k * 4u;
				xMat[r][off + 0u] = kLigandCharge[k];
				xMat[r][off + 1u] = bRot.i;
				xMat[r][off + 2u] = bRot.j;
				xMat[r][off + 3u] = bRot.k;
			}

			yMat[r][0] = (cls == 0u) ? 1.0f : 0.0f;
			yMat[r][1] = (cls == 1u) ? 1.0f : 0.0f;
			yMat[r][2] = (cls == 2u) ? 1.0f : 0.0f;
		}
	}

	featureCountCached = 12u;
	expectedCountCached = 3u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("Planar");
	classLabels.push_back("R-Enantiomer");
	classLabels.push_back("S-Enantiomer");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadSlaterKosterOrbital(unsigned int nTrain,
                                              unsigned int nTest,
                                              unsigned int seed,
                                              bool oodOrientation,
                                              QuaternionInputPackingMode packing,
                                              QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadSlaterKosterOrbital: nTrain and nTest must be > 0");
		return false;
	}

	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(12u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(1u, 0.0f));
	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(12u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(1u, 0.0f));

	for (unsigned int split = 0u; split < 2u; ++split)
	{
		const bool isTrain = (split == 0u);
		const unsigned int nRows = isTrain ? nTrain : nTest;
		shmea::GMatrix& xMat = isTrain ? trainMatrix : testMatrix;
		shmea::GMatrix& yMat = isTrain ? trainExpectedMatrix : testExpectedMatrix;

		for (unsigned int r = 0u; r < nRows; ++r)
		{
			const float d = glades::rng::uniform_float(eng, 0.80f, 1.80f);
			const Quaternion rHat = oodOrientation
			    ? (isTrain ? sample_hemisphere_vec3(eng, true, 0.65f)
			               : sample_hemisphere_vec3(eng, false, 0.35f))
			    : sample_unit_vec3_pure(eng);
			const Quaternion bond(0.0f, rHat.i * d, rHat.j * d, rHat.k * d);
			const Quaternion p1 = sample_unit_vec3_pure(eng);
			const Quaternion p2 = sample_unit_vec3_pure(eng);

			const float p1DotR = pure_vec_dot(p1, rHat);
			const float p2DotR = pure_vec_dot(p2, rHat);
			const float p1DotP2 = pure_vec_dot(p1, p2);
			const float sigmaTerm = p1DotR * p2DotR;
			const float piTerm = p1DotP2 - sigmaTerm;

			const float vSigma = -1.2f * expf(-1.1f * (d - 1.0f));
			const float vPi = 0.6f * expf(-1.4f * (d - 1.0f));
			const float overlap = (vSigma * sigmaTerm) + (vPi * piTerm);

			xMat[r][0] = 0.0f;
			xMat[r][1] = bond.i;
			xMat[r][2] = bond.j;
			xMat[r][3] = bond.k;
			xMat[r][4] = 0.0f;
			xMat[r][5] = p1.i;
			xMat[r][6] = p1.j;
			xMat[r][7] = p1.k;
			xMat[r][8] = 0.0f;
			xMat[r][9] = p2.i;
			xMat[r][10] = p2.j;
			xMat[r][11] = p2.k;

			yMat[r][0] = overlap;
		}
	}

	featureCountCached = 12u;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadKeplerOrbit(unsigned int nTrain,
                                      unsigned int nTest,
                                      unsigned int seed,
                                      bool oodOrbitalPlane,
                                      QuaternionInputPackingMode packing,
                                      QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadKeplerOrbit: nTrain and nTest must be > 0");
		return false;
	}

	static const float kPi = 3.14159265358979323846f;
	static const float kTwoPi = 2.0f * kPi;
	static const float kDegToRad = kPi / 180.0f;

	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));

	for (unsigned int split = 0u; split < 2u; ++split)
	{
		const bool isTrain = (split == 0u);
		const unsigned int nRows = isTrain ? nTrain : nTest;
		shmea::GMatrix& xMat = isTrain ? trainMatrix : testMatrix;
		shmea::GMatrix& yMat = isTrain ? trainExpectedMatrix : testExpectedMatrix;

		for (unsigned int r = 0u; r < nRows; ++r)
		{
			const float a = glades::rng::uniform_float(eng, 1.0f, 1.8f);
			const float ecc = glades::rng::uniform_float(eng, 0.10f, 0.55f);
			const float nu = glades::rng::uniform_float(eng, 0.0f, kTwoPi);

			const float p = a * (1.0f - (ecc * ecc));
			const float rad = p / (1.0f + (ecc * cosf(nu)));
			const float vScale = sqrtf(1.0f / p);

			const Quaternion rPf(0.0f, rad * cosf(nu), rad * sinf(nu), 0.0f);
			const Quaternion vPf(0.0f, -vScale * sinf(nu), vScale * (ecc + cosf(nu)), 0.0f);

			const Quaternion uRot = oodOrbitalPlane
			    ? (isTrain ? sample_inclination_rotation(eng, 0.0f, 25.0f * kDegToRad)
			               : sample_inclination_rotation(eng, 65.0f * kDegToRad, 180.0f * kDegToRad))
			    : sample_unit_quaternion(eng);

			const Quaternion pos = Quaternion::rotateVec(uRot, rPf);
			const Quaternion vel = Quaternion::rotateVec(uRot, vPf);

			// Exact conserved Keplerian quantities (mu = 1):
			//   eps = 0.5 * ||v||^2 - 1 / ||r|| = -1 / (2a)
			//   e   = v x (r x v) - r / ||r||
			const float posNorm = pos.norm();
			const float velNormSq = vel.normSq();
			const float energy = (0.5f * velNormSq) - (1.0f / posNorm);
			const Quaternion angMom = pure_vec_cross(pos, vel);
			const Quaternion vCrossL = pure_vec_cross(vel, angMom);
			const Quaternion lrl(
			    0.0f,
			    vCrossL.i - (pos.i / posNorm),
			    vCrossL.j - (pos.j / posNorm),
			    vCrossL.k - (pos.k / posNorm));

			xMat[r][0] = 0.0f;
			xMat[r][1] = pos.i;
			xMat[r][2] = pos.j;
			xMat[r][3] = pos.k;
			xMat[r][4] = 0.0f;
			xMat[r][5] = vel.i;
			xMat[r][6] = vel.j;
			xMat[r][7] = vel.k;

			yMat[r][0] = energy;
			yMat[r][1] = lrl.i;
			yMat[r][2] = lrl.j;
			yMat[r][3] = lrl.k;
		}
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::getTrainQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTrain = getTrainSize();
	if (index >= nTrain || quatInputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatInputCount);
	if (offset + static_cast<size_t>(quatInputCount) > trainQuatInputs.size())
		return false;
	outData = &trainQuatInputs[offset];
	outCount = quatInputCount;
	return true;
}

bool QuaternionInput::getTrainQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTrain = getTrainSize();
	if (index >= nTrain || quatOutputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatOutputCount);
	if (offset + static_cast<size_t>(quatOutputCount) > trainQuatTargets.size())
		return false;
	outData = &trainQuatTargets[offset];
	outCount = quatOutputCount;
	return true;
}

bool QuaternionInput::getTestQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTest = getTestSize();
	if (index >= nTest || quatInputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatInputCount);
	if (offset + static_cast<size_t>(quatInputCount) > testQuatInputs.size())
		return false;
	outData = &testQuatInputs[offset];
	outCount = quatInputCount;
	return true;
}

bool QuaternionInput::getTestQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTest = getTestSize();
	if (index >= nTest || quatOutputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatOutputCount);
	if (offset + static_cast<size_t>(quatOutputCount) > testQuatTargets.size())
		return false;
	outData = &testQuatTargets[offset];
	outCount = quatOutputCount;
	return true;
}

} // namespace glades
