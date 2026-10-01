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

#include "mapped-dataset-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/DataObjects/MappedMatrix.h"
#include "../../../Backend/Machine Learning/DataObjects/MappedNumberInput.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"

#include "Backend/Database/GVector.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace {

static bool mkdir_if_missing(const std::string& path)
{
	if (path.empty())
		return false;
	// 0777 filtered by umask (matches other UTs).
	::mkdir(path.c_str(), 0777);
	return true;
}

static bool write_bytes(const std::string& path, const void* data, size_t n)
{
	std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
	if (!out)
		return false;
	if (n > 0 && data)
		out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
	return static_cast<bool>(out);
}

static bool write_zero_file(const std::string& path, size_t n)
{
	std::ofstream out(path.c_str(), std::ios::out | std::ios::binary);
	if (!out)
		return false;
	for (size_t i = 0; i < n; ++i)
	{
		const unsigned char z = 0;
		out.write(reinterpret_cast<const char*>(&z), 1);
	}
	return static_cast<bool>(out);
}

static void build_small_matrix(shmea::GMatrix& out, unsigned int R, unsigned int C, float base)
{
	out = shmea::GMatrix(R, shmea::GVector<float>(C, 0.0f));
	for (unsigned int r = 0; r < R; ++r)
		for (unsigned int c = 0; c < C; ++c)
			out[r][c] = base + static_cast<float>(r * 10u + c);
}

} // namespace

void MappedDatasetUnitTest()
{
	printf("============================================================\n");
	printf("Mapped Dataset Unit Test Suite (.gcol + mmap)\n");
	printf("============================================================\n");

	// Use unit-tests/database/ as scratch to match other suites.
	mkdir_if_missing("database");
	mkdir_if_missing("database/mapped_datasets");

	printf("-----------------------------------\n");
	printf("Mapped Test A (MappedFloatMatrix round-trip)\n");
	printf("-----------------------------------\n");
	{
		shmea::GMatrix m;
		build_small_matrix(m, /*R*/ 3u, /*C*/ 2u, /*base*/ 1.0f);

		std::string err;
		const std::string path = "database/mapped_datasets/ut_a.gcol";
		const bool okW = glades::MappedFloatMatrix::writeFromGMatrix(path, m, &err);
		G_assert(__FILE__, __LINE__, "==============Mapped::A writeFromGMatrix failed==============", okW);

		glades::MappedFloatMatrix mm;
		const bool okO = mm.openReadOnly(path, &err);
		G_assert(__FILE__, __LINE__, "==============Mapped::A openReadOnly failed==============", okO);
		G_assert(__FILE__, __LINE__, "==============Mapped::A rows mismatch==============", mm.rows() == 3ull);
		G_assert(__FILE__, __LINE__, "==============Mapped::A cols mismatch==============", mm.cols() == 2ull);

		const float* r0 = mm.rowPtr(0ull);
		const float* r2 = mm.rowPtr(2ull);
		G_assert(__FILE__, __LINE__, "==============Mapped::A rowPtr(0) null==============", r0 != NULL);
		G_assert(__FILE__, __LINE__, "==============Mapped::A rowPtr(2) null==============", r2 != NULL);
		if (r0 && r2)
		{
			G_assert(__FILE__, __LINE__, "==============Mapped::A value mismatch r0c0==============", r0[0] == m[0][0]);
			G_assert(__FILE__, __LINE__, "==============Mapped::A value mismatch r0c1==============", r0[1] == m[0][1]);
			G_assert(__FILE__, __LINE__, "==============Mapped::A value mismatch r2c0==============", r2[0] == m[2][0]);
			G_assert(__FILE__, __LINE__, "==============Mapped::A value mismatch r2c1==============", r2[1] == m[2][1]);
		}
		// Out-of-range should return NULL.
		G_assert(__FILE__, __LINE__, "==============Mapped::A rowPtr out-of-range not null==============", mm.rowPtr(999ull) == NULL);
	}

	printf("-----------------------------------\n");
	printf("Mapped Test B (Header corruption detection)\n");
	printf("-----------------------------------\n");
	{
		// Too small file (smaller than header).
		{
			const std::string path = "database/mapped_datasets/ut_b_small.gcol";
			G_assert(__FILE__, __LINE__, "==============Mapped::B small write failed==============", write_zero_file(path, 8));
			glades::MappedFloatMatrix mm;
			std::string err;
			const bool ok = mm.openReadOnly(path, &err);
			G_assert(__FILE__, __LINE__, "==============Mapped::B small should fail==============", !ok);
		}

		// Wrong magic.
		{
			unsigned char hdr[64];
			std::memset(hdr, 0, sizeof(hdr));
			// Put an incorrect magic string.
			const char* bad = "NOT_GCOL";
			std::memcpy(hdr, bad, std::strlen(bad));
			const std::string path = "database/mapped_datasets/ut_b_magic.gcol";
			G_assert(__FILE__, __LINE__, "==============Mapped::B magic write failed==============", write_bytes(path, hdr, sizeof(hdr)));
			glades::MappedFloatMatrix mm;
			std::string err;
			const bool ok = mm.openReadOnly(path, &err);
			G_assert(__FILE__, __LINE__, "==============Mapped::B magic should fail==============", !ok);
		}
	}

	printf("-----------------------------------\n");
	printf("Mapped Test C (MappedNumberInput train-only + train/test)\n");
	printf("-----------------------------------\n");
	{
		// Build train-only dataset dir.
		const std::string dir = "database/mapped_datasets/ut_c_trainonly";
		mkdir_if_missing(dir);

		shmea::GMatrix Xtr, Ytr;
		build_small_matrix(Xtr, /*R*/ 4u, /*C*/ 3u, /*base*/ 0.0f);
		build_small_matrix(Ytr, /*R*/ 4u, /*C*/ 2u, /*base*/ 100.0f);

		std::string err;
		G_assert(__FILE__, __LINE__, "==============Mapped::C write train.x failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/train.x.gcol", Xtr, &err));
		G_assert(__FILE__, __LINE__, "==============Mapped::C write train.y failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/train.y.gcol", Ytr, &err));

		glades::MappedNumberInput di;
		di.import(shmea::GString(dir.c_str()), 0);
		G_assert(__FILE__, __LINE__, "==============Mapped::C di loaded failed==============", di.loadedOk());
		G_assert(__FILE__, __LINE__, "==============Mapped::C train size mismatch==============", di.getTrainSize() == 4u);
		G_assert(__FILE__, __LINE__, "==============Mapped::C test size should be 0==============", di.getTestSize() == 0u);
		G_assert(__FILE__, __LINE__, "==============Mapped::C feature count mismatch==============", di.getFeatureCount() == 3u);

		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		G_assert(__FILE__, __LINE__, "==============Mapped::C train row view failed==============", di.getTrainRowView(2u, x, xN));
		G_assert(__FILE__, __LINE__, "==============Mapped::C train exp view failed==============", di.getTrainExpectedRowView(2u, y, yN));
		G_assert(__FILE__, __LINE__, "==============Mapped::C train row dims mismatch==============", xN == 3u);
		G_assert(__FILE__, __LINE__, "==============Mapped::C train exp dims mismatch==============", yN == 2u);
		if (x && y)
		{
			G_assert(__FILE__, __LINE__, "==============Mapped::C train row value mismatch==============", x[0] == Xtr[2][0]);
			G_assert(__FILE__, __LINE__, "==============Mapped::C train exp value mismatch==============", y[1] == Ytr[2][1]);
		}

		// Now add a test split and re-import.
		shmea::GMatrix Xte, Yte;
		build_small_matrix(Xte, /*R*/ 2u, /*C*/ 3u, /*base*/ 50.0f);
		build_small_matrix(Yte, /*R*/ 2u, /*C*/ 2u, /*base*/ 150.0f);
		G_assert(__FILE__, __LINE__, "==============Mapped::C write test.x failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/test.x.gcol", Xte, &err));
		G_assert(__FILE__, __LINE__, "==============Mapped::C write test.y failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/test.y.gcol", Yte, &err));

		glades::MappedNumberInput di2;
		di2.import(shmea::GString(dir.c_str()), 0);
		G_assert(__FILE__, __LINE__, "==============Mapped::C2 di loaded failed==============", di2.loadedOk());
		G_assert(__FILE__, __LINE__, "==============Mapped::C2 test size mismatch==============", di2.getTestSize() == 2u);
		const float* x2 = NULL;
		unsigned int x2N = 0u;
		G_assert(__FILE__, __LINE__, "==============Mapped::C2 test row view failed==============", di2.getTestRowView(1u, x2, x2N));
		G_assert(__FILE__, __LINE__, "==============Mapped::C2 test row dims mismatch==============", x2N == 3u);
		if (x2)
			G_assert(__FILE__, __LINE__, "==============Mapped::C2 test row value mismatch==============", x2[2] == Xte[1][2]);
	}

	printf("-----------------------------------\n");
	printf("Mapped Test D (NNetwork integration consumes MappedNumberInput)\n");
	printf("-----------------------------------\n");
	{
		const std::string dir = "database/mapped_datasets/ut_d_export";
		mkdir_if_missing(dir);
		std::string err;

		// Write a tiny train/test split directly to .gcol.
		shmea::GMatrix Xtr(3u, shmea::GVector<float>(2, 0.0f));
		shmea::GMatrix Ytr(3u, shmea::GVector<float>(1, 0.0f));
		for (unsigned int r = 0; r < 3u; ++r)
		{
			Xtr[r][0] = static_cast<float>(r);
			Xtr[r][1] = static_cast<float>(r + 10u);
			Ytr[r][0] = static_cast<float>(r + 100u);
		}
		shmea::GMatrix Xte(1u, shmea::GVector<float>(2, 0.0f));
		shmea::GMatrix Yte(1u, shmea::GVector<float>(1, 0.0f));
		Xte[0][0] = 9.0f;
		Xte[0][1] = 19.0f;
		Yte[0][0] = 109.0f;

		G_assert(__FILE__, __LINE__, "==============Mapped::D write train.x failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/train.x.gcol", Xtr, &err));
		G_assert(__FILE__, __LINE__, "==============Mapped::D write train.y failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/train.y.gcol", Ytr, &err));
		G_assert(__FILE__, __LINE__, "==============Mapped::D write test.x failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/test.x.gcol", Xte, &err));
		G_assert(__FILE__, __LINE__, "==============Mapped::D write test.y failed==============",
		         glades::MappedFloatMatrix::writeFromGMatrix(dir + "/test.y.gcol", Yte, &err));

		glades::MappedNumberInput mdi;
		mdi.import(shmea::GString(dir.c_str()), 0);
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi loaded failed==============", mdi.loadedOk());
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train size mismatch==============", mdi.getTrainSize() == 3u);
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi test size mismatch==============", mdi.getTestSize() == 1u);
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi feature count mismatch==============", mdi.getFeatureCount() == 2u);

		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train row view failed==============", mdi.getTrainRowView(1u, x, xN));
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train exp view failed==============", mdi.getTrainExpectedRowView(1u, y, yN));
		G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train dims mismatch==============", xN == 2u && yN == 1u);
		if (x && y)
		{
			G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train value mismatch x0==============", x[0] == Xtr[1][0]);
			G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train value mismatch x1==============", x[1] == Xtr[1][1]);
			G_assert(__FILE__, __LINE__, "==============Mapped::D mdi train value mismatch y==============", y[0] == Ytr[1][0]);
		}

		// Integration: run a trivial net.test() on MappedNumberInput to ensure the training loop
		// can consume the view APIs (zero-copy) without relying on GMatrix internals.
		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_mapped_input_smoke", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(123u);
		net.getTerminatorMutable().setEpoch(0);
		net.getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net.test(&mdi);
		G_assert(__FILE__, __LINE__, "==============Mapped::D net.test status failed==============", st.ok());

		delete info; // owns in/hidden/out
	}

	printf("\n============================================================\n");
}

