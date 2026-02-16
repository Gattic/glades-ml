// Unit tests for DDP (Distributed Data Parallel) components.
#include "ddp-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/ddp_comm.h"
#include "../../../Backend/Machine Learning/DataObjects/DDPDataInputWrapper.h"
#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"

#include <math.h>
#include <string.h>
#include <vector>
#include <set>

namespace {

// ---------------------------------------------------------------------------
// Stub DataInput for DDPDataInputWrapper tests
// ---------------------------------------------------------------------------
// Provides a simple in-memory dataset with configurable sequences.
// Each row is a single float equal to its row index (for easy verification).

class StubDataInput : public glades::DataInput
{
	std::vector<float> trainRows;
	std::vector<float> testRows;
	unsigned int featureCount;

public:
	StubDataInput(unsigned int nTrainRows, unsigned int nTestRows, unsigned int nFeatures = 1)
	    : featureCount(nFeatures)
	{
		trainRows.resize(nTrainRows);
		for (unsigned int i = 0; i < nTrainRows; ++i)
			trainRows[i] = static_cast<float>(i);

		testRows.resize(nTestRows);
		for (unsigned int i = 0; i < nTestRows; ++i)
			testRows[i] = static_cast<float>(i + 1000);
	}

	virtual void import(shmea::GString, int) {}
	virtual void import(const shmea::GTable&, int) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int idx) const
	{
		shmea::GVector<float> v;
		if (idx < trainRows.size())
			v.push_back(trainRows[idx]);
		return v;
	}

	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int idx) const
	{
		return getTrainRow(idx);
	}

	virtual shmea::GVector<float> getTestRow(unsigned int idx) const
	{
		shmea::GVector<float> v;
		if (idx < testRows.size())
			v.push_back(testRows[idx]);
		return v;
	}

	virtual shmea::GVector<float> getTestExpectedRow(unsigned int idx) const
	{
		return getTestRow(idx);
	}

	virtual bool getTrainRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
	{
		if (idx >= trainRows.size()) { outData = NULL; outSize = 0; return false; }
		outData = &trainRows[idx];
		outSize = 1;
		return true;
	}

	virtual bool getTrainExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
	{
		return getTrainRowView(idx, outData, outSize);
	}

	virtual bool getTestRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
	{
		if (idx >= testRows.size()) { outData = NULL; outSize = 0; return false; }
		outData = &testRows[idx];
		outSize = 1;
		return true;
	}

	virtual bool getTestExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
	{
		return getTestRowView(idx, outData, outSize);
	}

	virtual unsigned int getTrainSize() const { return static_cast<unsigned int>(trainRows.size()); }
	virtual unsigned int getTestSize() const { return static_cast<unsigned int>(testRows.size()); }
	virtual unsigned int getFeatureCount() const { return featureCount; }
	virtual int getType() const { return DataInput::CSV; }

	// Expose protected sequence interface for test setup.
	void addTrainSequence(unsigned int start, unsigned int length)
	{
		trainSequences.push_back(SequenceSpan(start, length));
	}

	void addTestSequence(unsigned int start, unsigned int length)
	{
		testSequences.push_back(SequenceSpan(start, length));
	}
};

static bool float_eq(float a, float b, float eps = 1e-6f)
{
	return fabsf(a - b) < eps;
}

// ===========================================================================
// WarmupConfig tests
// ===========================================================================

static void Warmup_Defaults()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig defaults\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	ASSERT("default type is WARMUP_NONE", w.type == glades::WarmupConfig::WARMUP_NONE);
	ASSERT("default warmupSteps is 0", w.warmupSteps == 0);
	ASSERT("default multiplier at step 0 is 1.0", float_eq(w.multiplier(0), 1.0f));
	ASSERT("default multiplier at step 100 is 1.0", float_eq(w.multiplier(100), 1.0f));
}

static void Warmup_None_AlwaysOne()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig NONE always returns 1.0\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	w.type = glades::WarmupConfig::WARMUP_NONE;
	w.warmupSteps = 1000; // should be ignored when type is NONE

	ASSERT("NONE step 0", float_eq(w.multiplier(0), 1.0f));
	ASSERT("NONE step 500", float_eq(w.multiplier(500), 1.0f));
	ASSERT("NONE step 999", float_eq(w.multiplier(999), 1.0f));
	ASSERT("NONE step 1000", float_eq(w.multiplier(1000), 1.0f));
}

static void Warmup_Linear_ZeroSteps()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig LINEAR with 0 warmupSteps\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	w.type = glades::WarmupConfig::WARMUP_LINEAR;
	w.warmupSteps = 0;

	ASSERT("linear 0 steps at step 0", float_eq(w.multiplier(0), 1.0f));
	ASSERT("linear 0 steps at step 1", float_eq(w.multiplier(1), 1.0f));
}

static void Warmup_Linear_NegativeSteps()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig LINEAR with negative warmupSteps\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	w.type = glades::WarmupConfig::WARMUP_LINEAR;
	w.warmupSteps = -5;

	ASSERT("linear neg steps at step 0", float_eq(w.multiplier(0), 1.0f));
	ASSERT("linear neg steps at step 10", float_eq(w.multiplier(10), 1.0f));
}

static void Warmup_Linear_Ramp()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig LINEAR ramp correctness\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	w.type = glades::WarmupConfig::WARMUP_LINEAR;
	w.warmupSteps = 100;

	// At step 0: 0/100 = 0.0
	ASSERT("linear step 0", float_eq(w.multiplier(0), 0.0f));
	// At step 1: 1/100 = 0.01
	ASSERT("linear step 1", float_eq(w.multiplier(1), 0.01f));
	// At step 25: 25/100 = 0.25
	ASSERT("linear step 25", float_eq(w.multiplier(25), 0.25f));
	// At step 50: 50/100 = 0.5
	ASSERT("linear step 50", float_eq(w.multiplier(50), 0.5f));
	// At step 75: 75/100 = 0.75
	ASSERT("linear step 75", float_eq(w.multiplier(75), 0.75f));
	// At step 99: 99/100 = 0.99
	ASSERT("linear step 99", float_eq(w.multiplier(99), 0.99f));
	// At step 100: >= warmupSteps => 1.0
	ASSERT("linear step 100 (boundary)", float_eq(w.multiplier(100), 1.0f));
	// Beyond warmup: always 1.0
	ASSERT("linear step 200 (past warmup)", float_eq(w.multiplier(200), 1.0f));
	ASSERT("linear step 10000", float_eq(w.multiplier(10000), 1.0f));
}

static void Warmup_Linear_SingleStep()
{
	printf("-----------------------------------\n");
	printf("WarmupConfig LINEAR with warmupSteps=1\n");
	printf("-----------------------------------\n");

	glades::WarmupConfig w;
	w.type = glades::WarmupConfig::WARMUP_LINEAR;
	w.warmupSteps = 1;

	// At step 0: 0/1 = 0.0
	ASSERT("single step at 0", float_eq(w.multiplier(0), 0.0f));
	// At step 1: >= warmupSteps => 1.0
	ASSERT("single step at 1", float_eq(w.multiplier(1), 1.0f));
	ASSERT("single step at 2", float_eq(w.multiplier(2), 1.0f));
}

// ===========================================================================
// WorkerAddress tests
// ===========================================================================

static void DDPConfig_WorkerAddress_Defaults()
{
	printf("-----------------------------------\n");
	printf("WorkerAddress default ctor\n");
	printf("-----------------------------------\n");

	glades::ddp::WorkerAddress wa;
	ASSERT("default host is empty", wa.host.empty());
	ASSERT("default port is 0", wa.port == 0);
}

static void DDPConfig_WorkerAddress_Construct()
{
	printf("-----------------------------------\n");
	printf("WorkerAddress parameterized ctor\n");
	printf("-----------------------------------\n");

	glades::ddp::WorkerAddress wa("10.0.0.1", 9200);
	ASSERT("host is 10.0.0.1", wa.host == "10.0.0.1");
	ASSERT("port is 9200", wa.port == 9200);
}

// ===========================================================================
// DDPConfig tests
// ===========================================================================

static void DDPConfig_Defaults()
{
	printf("-----------------------------------\n");
	printf("DDPConfig defaults\n");
	printf("-----------------------------------\n");

	glades::DDPConfig d;
	ASSERT("default enable is false", d.enable == false);
	ASSERT("default linearLRScaling is true", d.linearLRScaling == true);
	ASSERT("default rank is 0", d.rank == 0);
	ASSERT("default worldSize is 1", d.worldSize == 1);
	ASSERT("default rootPort is 9200", d.rootPort == 9200);
}

// ===========================================================================
// TrainingConfig integration
// ===========================================================================

static void TrainingConfig_DDPAndWarmupMembers()
{
	printf("-----------------------------------\n");
	printf("TrainingConfig has ddp and warmup members\n");
	printf("-----------------------------------\n");

	glades::TrainingConfig tc;

	// Warmup defaults through TrainingConfig
	ASSERT("tc warmup type", tc.warmup.type == glades::WarmupConfig::WARMUP_NONE);
	ASSERT("tc warmup steps", tc.warmup.warmupSteps == 0);

	// DDP defaults through TrainingConfig
	ASSERT("tc ddp enable", tc.ddp.enable == false);
	ASSERT("tc ddp worldSize", tc.ddp.worldSize == 1);
	ASSERT("tc ddp rank", tc.ddp.rank == 0);

	// Mutate and verify
	tc.warmup.type = glades::WarmupConfig::WARMUP_LINEAR;
	tc.warmup.warmupSteps = 500;
	ASSERT("tc warmup multiplier after set", float_eq(tc.warmup.multiplier(250), 0.5f));

	tc.ddp.enable = true;
	tc.ddp.worldSize = 4;
	tc.ddp.rank = 2;
	ASSERT("tc ddp enable after set", tc.ddp.enable == true);
	ASSERT("tc ddp worldSize after set", tc.ddp.worldSize == 4);
	ASSERT("tc ddp rank after set", tc.ddp.rank == 2);
}

// ===========================================================================
// ddp_comm no-op tests (DDP not initialized)
// ===========================================================================

static void DDPComm_Noop_WorldSize()
{
	printf("-----------------------------------\n");
	printf("ddp_comm no-op: worldSize/rank/isRoot\n");
	printf("-----------------------------------\n");

	ASSERT("uninitialized worldSize is 1", glades::ddp::worldSize() == 1);
	ASSERT("uninitialized rank is 0", glades::ddp::rank() == 0);
	ASSERT("uninitialized isRoot is true", glades::ddp::isRoot() == true);
}

static void DDPComm_Noop_AllReduce()
{
	printf("-----------------------------------\n");
	printf("ddp_comm no-op: allReduceSumInPlace\n");
	printf("-----------------------------------\n");

	// When DDP is not initialized, allReduceSumInPlace should be a no-op.
	// Data should remain unchanged.
	float fBuf[] = {1.0f, 2.0f, 3.0f, 4.0f};
	glades::ddp::allReduceSumInPlace(fBuf, 4);
	ASSERT("float noop [0]", float_eq(fBuf[0], 1.0f));
	ASSERT("float noop [1]", float_eq(fBuf[1], 2.0f));
	ASSERT("float noop [2]", float_eq(fBuf[2], 3.0f));
	ASSERT("float noop [3]", float_eq(fBuf[3], 4.0f));

	unsigned int uBuf[] = {10, 20, 30};
	glades::ddp::allReduceSumInPlace(uBuf, 3);
	ASSERT("uint noop [0]", uBuf[0] == 10);
	ASSERT("uint noop [1]", uBuf[1] == 20);
	ASSERT("uint noop [2]", uBuf[2] == 30);

	double dBuf[] = {1.5, 2.5};
	glades::ddp::allReduceSumInPlace(dBuf, 2);
	ASSERT("double noop [0]", dBuf[0] == 1.5);
	ASSERT("double noop [1]", dBuf[1] == 2.5);
}

static void DDPComm_Noop_Broadcast()
{
	printf("-----------------------------------\n");
	printf("ddp_comm no-op: broadcastFromRoot\n");
	printf("-----------------------------------\n");

	float buf[] = {42.0f, 43.0f};
	glades::ddp::broadcastFromRoot(buf, 2);
	ASSERT("broadcast noop [0]", float_eq(buf[0], 42.0f));
	ASSERT("broadcast noop [1]", float_eq(buf[1], 43.0f));
}

static void DDPComm_Noop_Barrier()
{
	printf("-----------------------------------\n");
	printf("ddp_comm no-op: barrier\n");
	printf("-----------------------------------\n");

	// Should simply return without blocking or crashing.
	glades::ddp::barrier();
	ASSERT("barrier noop returns", true);
}

// ===========================================================================
// ddp_comm init with address list tests
// ===========================================================================

static void DDPComm_Noop_InitWithAddresses()
{
	printf("-----------------------------------\n");
	printf("ddp_comm init with 1-entry address list (no-op)\n");
	printf("-----------------------------------\n");

	// Finalize any prior state first.
	glades::ddp::finalize();

	std::vector<glades::ddp::WorkerAddress> addrs;
	addrs.push_back(glades::ddp::WorkerAddress("127.0.0.1", 9200));

	// worldSize==1 => should be a no-op (no networking).
	glades::ddp::init(addrs, 0);
	ASSERT("ws1 addr init worldSize", glades::ddp::worldSize() == 1);
	ASSERT("ws1 addr init rank", glades::ddp::rank() == 0);
	ASSERT("ws1 addr init isRoot", glades::ddp::isRoot() == true);

	glades::ddp::finalize();
	ASSERT("after finalize worldSize", glades::ddp::worldSize() == 1);
}

static void DDPComm_Noop_InitWithEmptyAddresses()
{
	printf("-----------------------------------\n");
	printf("ddp_comm init with empty address list (no-op)\n");
	printf("-----------------------------------\n");

	glades::ddp::finalize();

	std::vector<glades::ddp::WorkerAddress> addrs;

	// Empty vector => degenerate no-op.
	glades::ddp::init(addrs, 0);
	ASSERT("empty addr init worldSize", glades::ddp::worldSize() == 1);
	ASSERT("empty addr init rank", glades::ddp::rank() == 0);
	ASSERT("empty addr init isRoot", glades::ddp::isRoot() == true);

	glades::ddp::finalize();
}

// ===========================================================================
// DDPDataInputWrapper tests
// ===========================================================================

static void DDPWrapper_WorldSize1_Passthrough()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper worldSize=1 passthrough\n");
	printf("-----------------------------------\n");

	StubDataInput inner(100, 50);

	// Configure 10 train sequences, 5 test sequences
	for (unsigned int i = 0; i < 10; ++i)
		inner.addTrainSequence(i * 10, 10);
	for (unsigned int i = 0; i < 5; ++i)
		inner.addTestSequence(i * 10, 10);

	// worldSize=1: wrapper should pass everything through
	glades::DDPDataInputWrapper wrapper(&inner, 0, 1);

	ASSERT("ws1 train seq count", wrapper.getTrainSequenceCount() == 10);
	ASSERT("ws1 test seq count", wrapper.getTestSequenceCount() == 5);
	ASSERT("ws1 train size", wrapper.getTrainSize() == 100);
	ASSERT("ws1 test size", wrapper.getTestSize() == 50);
	ASSERT("ws1 feature count", wrapper.getFeatureCount() == 1);

	// Sequence lengths should match inner
	for (unsigned int s = 0; s < 10; ++s)
		ASSERT("ws1 train seq length", wrapper.getTrainSequenceLength(s) == 10);
}

static void DDPWrapper_TwoWorkers_EvenSplit()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper 2 workers, even split\n");
	printf("-----------------------------------\n");

	StubDataInput inner(80, 40);

	// 8 train sequences of 10 rows each
	for (unsigned int i = 0; i < 8; ++i)
		inner.addTrainSequence(i * 10, 10);
	// 4 test sequences of 10 rows each
	for (unsigned int i = 0; i < 4; ++i)
		inner.addTestSequence(i * 10, 10);

	glades::DDPDataInputWrapper rank0(&inner, 0, 2);
	glades::DDPDataInputWrapper rank1(&inner, 1, 2);

	// Round-robin: rank0 gets seqs 0,2,4,6; rank1 gets seqs 1,3,5,7
	ASSERT("r0 train seq count", rank0.getTrainSequenceCount() == 4);
	ASSERT("r1 train seq count", rank1.getTrainSequenceCount() == 4);

	ASSERT("r0 test seq count", rank0.getTestSequenceCount() == 2);
	ASSERT("r1 test seq count", rank1.getTestSequenceCount() == 2);

	// Verify sequence lengths are all 10
	for (unsigned int s = 0; s < 4; ++s)
	{
		ASSERT("r0 train seq length", rank0.getTrainSequenceLength(s) == 10);
		ASSERT("r1 train seq length", rank1.getTrainSequenceLength(s) == 10);
	}

	// Verify rank0 gets even-indexed sequences (global 0,2,4,6)
	// Sequence row data: inner seq 0 starts at row 0, seq 2 at row 20, etc.
	// Rank0's local seq 0 = global seq 0 => row 0 => value 0.0
	shmea::GVector<float> row = rank0.getTrainSequenceRow(0, 0);
	ASSERT("r0 seq0 t0 value", row.size() == 1 && float_eq(row[0], 0.0f));

	// Rank0's local seq 1 = global seq 2 => row 20 => value 20.0
	row = rank0.getTrainSequenceRow(1, 0);
	ASSERT("r0 seq1 t0 value", row.size() == 1 && float_eq(row[0], 20.0f));

	// Rank1's local seq 0 = global seq 1 => row 10 => value 10.0
	row = rank1.getTrainSequenceRow(0, 0);
	ASSERT("r1 seq0 t0 value", row.size() == 1 && float_eq(row[0], 10.0f));

	// Rank1's local seq 1 = global seq 3 => row 30 => value 30.0
	row = rank1.getTrainSequenceRow(1, 0);
	ASSERT("r1 seq1 t0 value", row.size() == 1 && float_eq(row[0], 30.0f));
}

static void DDPWrapper_ThreeWorkers_UnevenSplit()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper 3 workers, uneven split\n");
	printf("-----------------------------------\n");

	StubDataInput inner(70, 50);

	// 7 train sequences (not evenly divisible by 3)
	for (unsigned int i = 0; i < 7; ++i)
		inner.addTrainSequence(i * 10, 10);
	// 5 test sequences (not evenly divisible by 3)
	for (unsigned int i = 0; i < 5; ++i)
		inner.addTestSequence(i * 10, 10);

	glades::DDPDataInputWrapper rank0(&inner, 0, 3);
	glades::DDPDataInputWrapper rank1(&inner, 1, 3);
	glades::DDPDataInputWrapper rank2(&inner, 2, 3);

	// Round-robin distribution of 7 seqs:
	// rank0: 0, 3, 6 => 3 seqs
	// rank1: 1, 4    => 2 seqs
	// rank2: 2, 5    => 2 seqs
	ASSERT("3w r0 train count", rank0.getTrainSequenceCount() == 3);
	ASSERT("3w r1 train count", rank1.getTrainSequenceCount() == 2);
	ASSERT("3w r2 train count", rank2.getTrainSequenceCount() == 2);

	// Total should equal original
	unsigned int totalTrain = rank0.getTrainSequenceCount()
	                        + rank1.getTrainSequenceCount()
	                        + rank2.getTrainSequenceCount();
	ASSERT("3w total train seqs", totalTrain == 7);

	// Round-robin distribution of 5 test seqs:
	// rank0: 0, 3    => 2 seqs
	// rank1: 1, 4    => 2 seqs
	// rank2: 2       => 1 seq
	ASSERT("3w r0 test count", rank0.getTestSequenceCount() == 2);
	ASSERT("3w r1 test count", rank1.getTestSequenceCount() == 2);
	ASSERT("3w r2 test count", rank2.getTestSequenceCount() == 1);

	unsigned int totalTest = rank0.getTestSequenceCount()
	                       + rank1.getTestSequenceCount()
	                       + rank2.getTestSequenceCount();
	ASSERT("3w total test seqs", totalTest == 5);
}

static void DDPWrapper_NoOverlap()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper no sequence overlap\n");
	printf("-----------------------------------\n");

	StubDataInput inner(120, 0);

	// 12 train sequences of 10 rows each
	for (unsigned int i = 0; i < 12; ++i)
		inner.addTrainSequence(i * 10, 10);

	const int ws = 4;
	std::set<unsigned int> allFirstRows;
	unsigned int totalSeqs = 0;

	for (int r = 0; r < ws; ++r)
	{
		glades::DDPDataInputWrapper wrapper(&inner, r, ws);
		unsigned int count = wrapper.getTrainSequenceCount();
		totalSeqs += count;
		for (unsigned int s = 0; s < count; ++s)
		{
			// First row of each sequence should be unique across workers
			shmea::GVector<float> row = wrapper.getTrainSequenceRow(s, 0);
			ASSERT("row has data", row.size() == 1);
			unsigned int rowVal = static_cast<unsigned int>(row[0]);
			ASSERT("no duplicate first row", allFirstRows.find(rowVal) == allFirstRows.end());
			allFirstRows.insert(rowVal);
		}
	}

	ASSERT("total seqs covers all", totalSeqs == 12);
	ASSERT("all first rows accounted for", allFirstRows.size() == 12);
}

static void DDPWrapper_VariableLengthSequences()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper variable-length sequences\n");
	printf("-----------------------------------\n");

	// 5 sequences of lengths 3, 7, 2, 8, 5 = 25 total rows
	StubDataInput inner(25, 0);
	inner.addTrainSequence(0, 3);
	inner.addTrainSequence(3, 7);
	inner.addTrainSequence(10, 2);
	inner.addTrainSequence(12, 8);
	inner.addTrainSequence(20, 5);

	glades::DDPDataInputWrapper rank0(&inner, 0, 2);
	glades::DDPDataInputWrapper rank1(&inner, 1, 2);

	// rank0: seqs 0, 2, 4 (lengths 3, 2, 5)
	// rank1: seqs 1, 3    (lengths 7, 8)
	ASSERT("varlen r0 count", rank0.getTrainSequenceCount() == 3);
	ASSERT("varlen r1 count", rank1.getTrainSequenceCount() == 2);

	ASSERT("varlen r0 seq0 len", rank0.getTrainSequenceLength(0) == 3);
	ASSERT("varlen r0 seq1 len", rank0.getTrainSequenceLength(1) == 2);
	ASSERT("varlen r0 seq2 len", rank0.getTrainSequenceLength(2) == 5);

	ASSERT("varlen r1 seq0 len", rank1.getTrainSequenceLength(0) == 7);
	ASSERT("varlen r1 seq1 len", rank1.getTrainSequenceLength(1) == 8);

	// Verify data: rank0 seq1 = global seq 2 (start=10, len=2)
	// t=0 => row 10 => value 10.0, t=1 => row 11 => value 11.0
	shmea::GVector<float> row = rank0.getTrainSequenceRow(1, 0);
	ASSERT("varlen r0 seq1 t0", row.size() == 1 && float_eq(row[0], 10.0f));
	row = rank0.getTrainSequenceRow(1, 1);
	ASSERT("varlen r0 seq1 t1", row.size() == 1 && float_eq(row[0], 11.0f));

	// rank1 seq1 = global seq 3 (start=12, len=8)
	// t=0 => row 12 => value 12.0
	row = rank1.getTrainSequenceRow(1, 0);
	ASSERT("varlen r1 seq1 t0", row.size() == 1 && float_eq(row[0], 12.0f));
	// t=7 (last) => row 19 => value 19.0
	row = rank1.getTrainSequenceRow(1, 7);
	ASSERT("varlen r1 seq1 t7", row.size() == 1 && float_eq(row[0], 19.0f));
}

static void DDPWrapper_OutOfBounds()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper out-of-bounds access\n");
	printf("-----------------------------------\n");

	StubDataInput inner(30, 0);
	for (unsigned int i = 0; i < 3; ++i)
		inner.addTrainSequence(i * 10, 10);

	glades::DDPDataInputWrapper wrapper(&inner, 0, 2);
	// rank0 gets seqs 0, 2 => 2 local seqs

	ASSERT("oob seq count", wrapper.getTrainSequenceCount() == 2);
	ASSERT("oob seq 99 length", wrapper.getTrainSequenceLength(99) == 0);

	shmea::GVector<float> row = wrapper.getTrainSequenceRow(99, 0);
	ASSERT("oob seq 99 row empty", row.size() == 0);
}

static void DDPWrapper_NullInner()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper NULL inner\n");
	printf("-----------------------------------\n");

	glades::DDPDataInputWrapper wrapper(NULL, 0, 2);

	ASSERT("null train seq count", wrapper.getTrainSequenceCount() == 0);
	ASSERT("null test seq count", wrapper.getTestSequenceCount() == 0);
	ASSERT("null train size", wrapper.getTrainSize() == 0);
	ASSERT("null test size", wrapper.getTestSize() == 0);
	ASSERT("null feature count", wrapper.getFeatureCount() == 0);
}

static void DDPWrapper_SingleSequence()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper single-sequence dataset\n");
	printf("-----------------------------------\n");

	// No explicit sequences: DataInput treats entire dataset as 1 sequence.
	StubDataInput inner(50, 0);
	// Don't add any sequences - default behavior

	glades::DDPDataInputWrapper wrapper(&inner, 0, 2);

	// With worldSize=2 but only 1 default sequence,
	// round-robin: seq 0 % 2 == 0 => rank 0 gets it, rank 1 gets nothing
	ASSERT("single seq r0 count", wrapper.getTrainSequenceCount() == 1);

	glades::DDPDataInputWrapper wrapper1(&inner, 1, 2);
	ASSERT("single seq r1 count", wrapper1.getTrainSequenceCount() == 0);
}

static void DDPWrapper_ManyWorkersFewerSequences()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper more workers than sequences\n");
	printf("-----------------------------------\n");

	StubDataInput inner(20, 0);
	inner.addTrainSequence(0, 10);
	inner.addTrainSequence(10, 10);

	// 2 sequences, 5 workers: ranks 0,1 get 1 each, ranks 2,3,4 get 0
	unsigned int total = 0;
	for (int r = 0; r < 5; ++r)
	{
		glades::DDPDataInputWrapper wrapper(&inner, r, 5);
		unsigned int count = wrapper.getTrainSequenceCount();
		total += count;
		if (r < 2)
			ASSERT("worker with seq", count == 1);
		else
			ASSERT("worker without seq", count == 0);
	}
	ASSERT("total equals 2", total == 2);
}

static void DDPWrapper_DelegatesRowAccess()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper delegates row access\n");
	printf("-----------------------------------\n");

	StubDataInput inner(10, 5);

	glades::DDPDataInputWrapper wrapper(&inner, 0, 2);

	// Row-level access delegates directly (no sequence mapping)
	shmea::GVector<float> row = wrapper.getTrainRow(3);
	ASSERT("delegate train row", row.size() == 1 && float_eq(row[0], 3.0f));

	row = wrapper.getTestRow(2);
	ASSERT("delegate test row", row.size() == 1 && float_eq(row[0], 1002.0f));

	// View-based access
	const float* data = NULL;
	unsigned int sz = 0;
	bool ok = wrapper.getTrainRowView(5, data, sz);
	ASSERT("delegate train view ok", ok);
	ASSERT("delegate train view size", sz == 1);
	ASSERT("delegate train view value", data != NULL && float_eq(*data, 5.0f));

	ASSERT("delegate getType", wrapper.getType() == glades::DataInput::CSV);
}

static void DDPWrapper_TestSequencePartitioning()
{
	printf("-----------------------------------\n");
	printf("DDPDataInputWrapper test seq partitioning\n");
	printf("-----------------------------------\n");

	StubDataInput inner(0, 60);

	// 6 test sequences of 10 rows each
	for (unsigned int i = 0; i < 6; ++i)
		inner.addTestSequence(i * 10, 10);

	glades::DDPDataInputWrapper rank0(&inner, 0, 3);
	glades::DDPDataInputWrapper rank1(&inner, 1, 3);
	glades::DDPDataInputWrapper rank2(&inner, 2, 3);

	// rank0: seqs 0, 3 => 2 seqs
	// rank1: seqs 1, 4 => 2 seqs
	// rank2: seqs 2, 5 => 2 seqs
	ASSERT("test r0 count", rank0.getTestSequenceCount() == 2);
	ASSERT("test r1 count", rank1.getTestSequenceCount() == 2);
	ASSERT("test r2 count", rank2.getTestSequenceCount() == 2);

	// rank0 seq1 = global seq 3, start=30
	// t=0 => test row 30 => value 1030.0
	shmea::GVector<float> row = rank0.getTestSequenceRow(1, 0);
	ASSERT("test r0 seq1 t0", row.size() == 1 && float_eq(row[0], 1030.0f));

	// rank2 seq0 = global seq 2, start=20
	row = rank2.getTestSequenceRow(0, 0);
	ASSERT("test r2 seq0 t0", row.size() == 1 && float_eq(row[0], 1020.0f));
}

} // namespace

void DDPUnitTest()
{
	printf("============================================================\n");
	printf("DDP Unit Tests\n");
	printf("============================================================\n");

	// WarmupConfig
	Warmup_Defaults();
	Warmup_None_AlwaysOne();
	Warmup_Linear_ZeroSteps();
	Warmup_Linear_NegativeSteps();
	Warmup_Linear_Ramp();
	Warmup_Linear_SingleStep();

	// WorkerAddress
	DDPConfig_WorkerAddress_Defaults();
	DDPConfig_WorkerAddress_Construct();

	// DDPConfig
	DDPConfig_Defaults();

	// TrainingConfig integration
	TrainingConfig_DDPAndWarmupMembers();

	// ddp_comm no-ops
	DDPComm_Noop_WorldSize();
	DDPComm_Noop_AllReduce();
	DDPComm_Noop_Broadcast();
	DDPComm_Noop_Barrier();

	// ddp_comm init with address list
	DDPComm_Noop_InitWithAddresses();
	DDPComm_Noop_InitWithEmptyAddresses();

	// DDPDataInputWrapper
	DDPWrapper_WorldSize1_Passthrough();
	DDPWrapper_TwoWorkers_EvenSplit();
	DDPWrapper_ThreeWorkers_UnevenSplit();
	DDPWrapper_NoOverlap();
	DDPWrapper_VariableLengthSequences();
	DDPWrapper_OutOfBounds();
	DDPWrapper_NullInner();
	DDPWrapper_SingleSequence();
	DDPWrapper_ManyWorkersFewerSequences();
	DDPWrapper_DelegatesRowAccess();
	DDPWrapper_TestSequencePartitioning();

	printf("\n============================================================\n");
}
