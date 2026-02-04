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
#ifndef _GDATAINPUT
#define _GDATAINPUT

#include "Backend/Database/GTable.h"
#include "Backend/Database/GPointer.h"
#include "../GMath/OHE.h"
#include <vector>
#include <limits>
#include <string>
#include <stdio.h> // sprintf

namespace glades {

class DataInput
{
public:
	// Forward-declare so we can store spans internally.
	struct SequenceSpan;

protected:

    float min;
    float max;

	// Explicit sequence spans; if empty, default single-sequence behavior applies.
	std::vector<SequenceSpan> trainSequences;
	std::vector<SequenceSpan> testSequences;

public:

	const static int CSV = 0;
	const static int IMAGE = 1;
	const static int TEXT = 2;

	// === Sequence modeling ===
	//
	// Many datasets are best represented as multiple independent sequences
	// (e.g. sentences, trajectories, time series per entity).
	//
	// A sequence is represented as a contiguous span of the underlying row storage:
	// rows [start, start+length).
	struct SequenceSpan
	{
		unsigned int start;
		unsigned int length;
		SequenceSpan() : start(0u), length(0u) {}
		SequenceSpan(unsigned int s, unsigned int l) : start(s), length(l) {}
	};

	DataInput()
	{
	    min = std::numeric_limits<float>::max();
	    // NOTE: numeric_limits<float>::min() is the smallest *positive* normal float,
	    // not the most negative value. Use a portable C++98-friendly initialization.
	    max = -std::numeric_limits<float>::max();
	}

	virtual ~DataInput()
    {
        //
    }

	std::vector<shmea::GPointer<OHE> > trainingOHEMaps;
	std::vector<bool> trainingFeatureIsCategorical;
	std::vector<shmea::GPointer<OHE> > testingOHEMaps;
	std::vector<bool> testingFeatureIsCategorical;

	virtual void import(shmea::GString, int = 0) = 0;
	virtual void import(const shmea::GTable&, int = 0) = 0;

	virtual shmea::GVector<float> getTrainRow(unsigned int) const = 0;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int) const = 0;

	virtual shmea::GVector<float> getTestRow(unsigned int) const = 0;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int) const = 0;

	// === Zero-copy row access (hot path) ===
	//
	// These APIs allow the training loop to read feature/expected rows without forcing
	// a per-call allocation/copy of a shmea::GVector<float>.
	//
	// Lifetime:
	// - Returned pointers remain valid until the next call that may mutate the underlying
	//   storage for this DataInput instance (e.g., cache insert/eviction in ImageInput),
	//   or until the DataInput is destroyed.
	//
	// Thread-safety:
	// - Implementations may use mutable caches under this const API; do not assume
	//   thread safety unless the specific DataInput documents it.
	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const = 0;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const = 0;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const = 0;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const = 0;

	virtual unsigned int getTrainSize() const = 0;
	virtual unsigned int getTestSize() const = 0;
	virtual unsigned int getFeatureCount() const = 0;

	// === Shape contract (fast validation) ===
	//
	// TrainingCore previously validated shapes by materializing *every* train row and expected row.
	// That is catastrophically expensive for streaming inputs (e.g. ImageInput where getTrainRow()
	// loads/decodes images from disk).
	//
	// DataInput implementations should override these methods when row shapes are fixed and known
	// without materializing rows.
	//
	// Semantics:
	// - "fixed row size" means all train rows have identical dimensionality.
	// - returning true implies getFixed*Size() returns the dimensionality (>0).
	virtual bool hasFixedTrainRowSize() const { return false; }
	virtual unsigned int getFixedTrainRowSize() const { return 0u; }
	virtual bool hasFixedTrainExpectedRowSize() const { return false; }
	virtual unsigned int getFixedTrainExpectedRowSize() const { return 0u; }

	// Validate that (train row size >= expectedFeatureCount) and
	// (expected row size >= expectedOutSize).
	//
	// If fixed-size info is available, this is O(1). Otherwise, it checks a bounded set of rows
	// (defaults to up to 8 indices spanning the dataset) to avoid O(N) dataset materialization.
	bool validateTrainRowShapes(unsigned int expectedFeatureCount,
	                            unsigned int expectedOutSize,
	                            std::string* errMsg = NULL,
	                            unsigned int maxRowsToCheck = 8u) const;

	// Test-set analogue of validateTrainRowShapes().
	//
	// IMPORTANT: this performs a bounded spot-check (up to maxRowsToCheck) to avoid
	// materializing the full dataset for streaming inputs.
	bool validateTestRowShapes(unsigned int expectedFeatureCount,
	                           unsigned int expectedOutSize,
	                           std::string* errMsg = NULL,
	                           unsigned int maxRowsToCheck = 8u) const;

	// === Sequence interface (used by RNN/GRU/LSTM code paths) ===
	//
	// Default behavior (compatibility): if no explicit sequences were configured,
	// treat the entire train/test set as a single sequence whose timesteps
	// correspond to rows 0..N-1.
	//
	// IMPORTANT: This interface is intentionally row-based (not returning references)
	// to preserve existing DataInput implementations.
	virtual unsigned int getTrainSequenceCount() const;
	virtual unsigned int getTrainSequenceLength(unsigned int seqIdx) const;
	virtual shmea::GVector<float> getTrainSequenceRow(unsigned int seqIdx, unsigned int t) const;
	virtual shmea::GVector<float> getTrainSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const;

	virtual unsigned int getTestSequenceCount() const;
	virtual unsigned int getTestSequenceLength(unsigned int seqIdx) const;
	virtual shmea::GVector<float> getTestSequenceRow(unsigned int seqIdx, unsigned int t) const;
	virtual shmea::GVector<float> getTestSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const;

	// Zero-copy variants of the sequence getters (implemented in terms of row views + SequenceSpan mapping).
	bool getTrainSequenceRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const;
	bool getTrainSequenceExpectedRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const;
	bool getTestSequenceRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const;
	bool getTestSequenceExpectedRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const;

	// Configure explicit sequence spans. Returns false if spans are invalid *given
	// the currently-loaded dataset sizes*. If called before import()/data load,
	// spans are stored but validation will be deferred until validate*Sequences().
	bool setTrainSequences(const std::vector<SequenceSpan>& spans);
	bool setTestSequences(const std::vector<SequenceSpan>& spans);
	void clearTrainSequences();
	void clearTestSequences();

	// Convenience: configure sequences by their start indices (must be sorted, unique).
	// Example: starts=[0, 10, 25] => sequences [0..9], [10..24], [25..N-1].
	bool setTrainSequenceStarts(const std::vector<unsigned int>& starts);
	bool setTestSequenceStarts(const std::vector<unsigned int>& starts);

	bool validateTrainSequences(std::string* errMsg = NULL) const;
	bool validateTestSequences(std::string* errMsg = NULL) const;

	float getMin() const;
	float getMax() const;
	float getRange() const;

	virtual int getType() const = 0;
};

// ===== Inline sequence-model implementation =====
// Keep these in the header so unit-tests that link against an installed libglades
// still get the updated sequence semantics when building from this repo's headers.

inline unsigned int DataInput::getTrainSequenceCount() const
{
	if (!trainSequences.empty())
		return static_cast<unsigned int>(trainSequences.size());
	return (getTrainSize() > 0 ? 1u : 0u);
}

inline unsigned int DataInput::getTrainSequenceLength(unsigned int seqIdx) const
{
	if (!trainSequences.empty())
	{
		if (seqIdx >= trainSequences.size())
			return 0u;
		return trainSequences[seqIdx].length;
	}
	return (seqIdx == 0 ? getTrainSize() : 0u);
}

inline shmea::GVector<float> DataInput::getTrainSequenceRow(unsigned int seqIdx, unsigned int t) const
{
	if (!trainSequences.empty())
	{
		if (seqIdx >= trainSequences.size())
			return shmea::GVector<float>();
		const SequenceSpan& s = trainSequences[seqIdx];
		if (t >= s.length)
			return shmea::GVector<float>();
		return getTrainRow(s.start + t);
	}
	return getTrainRow(t);
}

inline shmea::GVector<float> DataInput::getTrainSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const
{
	if (!trainSequences.empty())
	{
		if (seqIdx >= trainSequences.size())
			return shmea::GVector<float>();
		const SequenceSpan& s = trainSequences[seqIdx];
		if (t >= s.length)
			return shmea::GVector<float>();
		return getTrainExpectedRow(s.start + t);
	}
	return getTrainExpectedRow(t);
}

inline unsigned int DataInput::getTestSequenceCount() const
{
	if (!testSequences.empty())
		return static_cast<unsigned int>(testSequences.size());
	return (getTestSize() > 0 ? 1u : 0u);
}

inline unsigned int DataInput::getTestSequenceLength(unsigned int seqIdx) const
{
	if (!testSequences.empty())
	{
		if (seqIdx >= testSequences.size())
			return 0u;
		return testSequences[seqIdx].length;
	}
	return (seqIdx == 0 ? getTestSize() : 0u);
}

inline shmea::GVector<float> DataInput::getTestSequenceRow(unsigned int seqIdx, unsigned int t) const
{
	if (!testSequences.empty())
	{
		if (seqIdx >= testSequences.size())
			return shmea::GVector<float>();
		const SequenceSpan& s = testSequences[seqIdx];
		if (t >= s.length)
			return shmea::GVector<float>();
		return getTestRow(s.start + t);
	}
	return getTestRow(t);
}

inline shmea::GVector<float> DataInput::getTestSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const
{
	if (!testSequences.empty())
	{
		if (seqIdx >= testSequences.size())
			return shmea::GVector<float>();
		const SequenceSpan& s = testSequences[seqIdx];
		if (t >= s.length)
			return shmea::GVector<float>();
		return getTestExpectedRow(s.start + t);
	}
	return getTestExpectedRow(t);
}

inline bool DataInput::getTrainSequenceRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!trainSequences.empty())
	{
		if (seqIdx >= trainSequences.size())
			return false;
		const SequenceSpan& s = trainSequences[seqIdx];
		if (t >= s.length)
			return false;
		return getTrainRowView(s.start + t, outData, outSize);
	}
	return getTrainRowView(t, outData, outSize);
}

inline bool DataInput::getTrainSequenceExpectedRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!trainSequences.empty())
	{
		if (seqIdx >= trainSequences.size())
			return false;
		const SequenceSpan& s = trainSequences[seqIdx];
		if (t >= s.length)
			return false;
		return getTrainExpectedRowView(s.start + t, outData, outSize);
	}
	return getTrainExpectedRowView(t, outData, outSize);
}

inline bool DataInput::getTestSequenceRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!testSequences.empty())
	{
		if (seqIdx >= testSequences.size())
			return false;
		const SequenceSpan& s = testSequences[seqIdx];
		if (t >= s.length)
			return false;
		return getTestRowView(s.start + t, outData, outSize);
	}
	return getTestRowView(t, outData, outSize);
}

inline bool DataInput::getTestSequenceExpectedRowView(unsigned int seqIdx, unsigned int t, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (!testSequences.empty())
	{
		if (seqIdx >= testSequences.size())
			return false;
		const SequenceSpan& s = testSequences[seqIdx];
		if (t >= s.length)
			return false;
		return getTestExpectedRowView(s.start + t, outData, outSize);
	}
	return getTestExpectedRowView(t, outData, outSize);
}

inline void DataInput::clearTrainSequences()
{
	trainSequences.clear();
}

inline void DataInput::clearTestSequences()
{
	testSequences.clear();
}

inline bool DataInput::validateTrainSequences(std::string* errMsg) const
{
	if (trainSequences.empty())
		return true;

	const unsigned int totalRows = getTrainSize();
	if (totalRows == 0)
	{
		if (errMsg) *errMsg = "sequence spans configured but dataset size is 0";
		return false;
	}

	unsigned int prevEnd = 0u;
	for (unsigned int i = 0; i < trainSequences.size(); ++i)
	{
		const unsigned int start = trainSequences[i].start;
		const unsigned int len = trainSequences[i].length;
		if (len == 0u)
		{
			if (errMsg) *errMsg = "sequence span has length 0";
			return false;
		}
		if (start >= totalRows)
		{
			if (errMsg) *errMsg = "sequence span start out of bounds";
			return false;
		}
		if (start + len > totalRows)
		{
			if (errMsg) *errMsg = "sequence span end out of bounds";
			return false;
		}
		if (i > 0u && start < prevEnd)
		{
			if (errMsg) *errMsg = "sequence spans overlap or are not sorted";
			return false;
		}
		prevEnd = start + len;
	}
	return true;
}

inline bool DataInput::validateTestSequences(std::string* errMsg) const
{
	if (testSequences.empty())
		return true;

	const unsigned int totalRows = getTestSize();
	if (totalRows == 0)
	{
		if (errMsg) *errMsg = "sequence spans configured but dataset size is 0";
		return false;
	}

	unsigned int prevEnd = 0u;
	for (unsigned int i = 0; i < testSequences.size(); ++i)
	{
		const unsigned int start = testSequences[i].start;
		const unsigned int len = testSequences[i].length;
		if (len == 0u)
		{
			if (errMsg) *errMsg = "sequence span has length 0";
			return false;
		}
		if (start >= totalRows)
		{
			if (errMsg) *errMsg = "sequence span start out of bounds";
			return false;
		}
		if (start + len > totalRows)
		{
			if (errMsg) *errMsg = "sequence span end out of bounds";
			return false;
		}
		if (i > 0u && start < prevEnd)
		{
			if (errMsg) *errMsg = "sequence spans overlap or are not sorted";
			return false;
		}
		prevEnd = start + len;
	}
	return true;
}

inline bool DataInput::setTrainSequences(const std::vector<SequenceSpan>& spans)
{
	trainSequences = spans;
	// Validate only if we have data; otherwise defer.
	if (getTrainSize() == 0)
		return true;
	return validateTrainSequences(NULL);
}

inline bool DataInput::setTestSequences(const std::vector<SequenceSpan>& spans)
{
	testSequences = spans;
	if (getTestSize() == 0)
		return true;
	return validateTestSequences(NULL);
}

inline bool DataInput::setTrainSequenceStarts(const std::vector<unsigned int>& starts)
{
	trainSequences.clear();
	const unsigned int totalRows = getTrainSize();
	for (unsigned int i = 0; i < starts.size(); ++i)
	{
		const unsigned int s = starts[i];
		const unsigned int end = (i + 1 < starts.size()) ? starts[i + 1] : totalRows;
		const unsigned int len = (end >= s) ? (end - s) : 0u;
		trainSequences.push_back(SequenceSpan(s, len));
	}
	if (getTrainSize() == 0)
		return true;
	return validateTrainSequences(NULL);
}

inline bool DataInput::setTestSequenceStarts(const std::vector<unsigned int>& starts)
{
	testSequences.clear();
	const unsigned int totalRows = getTestSize();
	for (unsigned int i = 0; i < starts.size(); ++i)
	{
		const unsigned int s = starts[i];
		const unsigned int end = (i + 1 < starts.size()) ? starts[i + 1] : totalRows;
		const unsigned int len = (end >= s) ? (end - s) : 0u;
		testSequences.push_back(SequenceSpan(s, len));
	}
	if (getTestSize() == 0)
		return true;
	return validateTestSequences(NULL);
}

inline bool DataInput::validateTrainRowShapes(unsigned int expectedFeatureCount,
                                              unsigned int expectedOutSize,
                                              std::string* errMsg,
                                              unsigned int maxRowsToCheck) const
{
	if (expectedFeatureCount == 0u)
	{
		if (errMsg) *errMsg = "expectedFeatureCount is 0";
		return false;
	}
	if (expectedOutSize == 0u)
	{
		if (errMsg) *errMsg = "expectedOutSize is 0";
		return false;
	}

	const unsigned int trainSize = getTrainSize();
	if (trainSize == 0u)
	{
		// TrainingCore already treats this as an error; keep this helper permissive.
		return true;
	}

	// O(1) validation for fixed-shape inputs.
	if (hasFixedTrainRowSize())
	{
		const unsigned int n = getFixedTrainRowSize();
		if (n < expectedFeatureCount)
		{
			if (errMsg)
			{
				char buf[256];
				sprintf(buf,
				        "train feature count (%u) is smaller than expectedFeatureCount (%u)",
				        n, expectedFeatureCount);
				*errMsg = std::string(buf);
			}
			return false;
		}
	}
	if (hasFixedTrainExpectedRowSize())
	{
		const unsigned int n = getFixedTrainExpectedRowSize();
		if (n < expectedOutSize)
		{
			if (errMsg)
			{
				char buf[256];
				sprintf(buf,
				        "train expected output count (%u) is smaller than expectedOutSize (%u)",
				        n, expectedOutSize);
				*errMsg = std::string(buf);
			}
			return false;
		}
	}

	// If either fixed-size contract is missing, do a bounded materialization check.
	if (!(hasFixedTrainRowSize() && hasFixedTrainExpectedRowSize()))
	{
		const unsigned int wantChecks = (maxRowsToCheck == 0u ? 1u : maxRowsToCheck);
		const unsigned int checks = (trainSize < wantChecks ? trainSize : wantChecks);
		const unsigned int denom = (checks > 1u ? (checks - 1u) : 1u);

		for (unsigned int k = 0; k < checks; ++k)
		{
			const unsigned int idx = (trainSize == 1u) ? 0u : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(trainSize - 1u)) / denom);

			const shmea::GVector<float> row = getTrainRow(idx);
			if (row.size() < expectedFeatureCount)
			{
				if (errMsg)
				{
					char buf[256];
					sprintf(buf,
					        "training row %u has %u features but expected at least %u",
					        idx,
					        static_cast<unsigned int>(row.size()),
					        expectedFeatureCount);
					*errMsg = std::string(buf);
				}
				return false;
			}

			const shmea::GVector<float> exp = getTrainExpectedRow(idx);
			if (exp.size() < expectedOutSize)
			{
				if (errMsg)
				{
					char buf[256];
					sprintf(buf,
					        "expected row %u has %u outputs but expected at least %u",
					        idx,
					        static_cast<unsigned int>(exp.size()),
					        expectedOutSize);
					*errMsg = std::string(buf);
				}
				return false;
			}
		}
	}

	return true;
}

inline bool DataInput::validateTestRowShapes(unsigned int expectedFeatureCount,
                                             unsigned int expectedOutSize,
                                             std::string* errMsg,
                                             unsigned int maxRowsToCheck) const
{
	if (expectedFeatureCount == 0u)
	{
		if (errMsg) *errMsg = "expectedFeatureCount is 0";
		return false;
	}
	if (expectedOutSize == 0u)
	{
		if (errMsg) *errMsg = "expectedOutSize is 0";
		return false;
	}

	const unsigned int testSize = getTestSize();
	if (testSize == 0u)
	{
		// Trainer already treats this as an error; keep this helper permissive.
		return true;
	}

	const unsigned int wantChecks = (maxRowsToCheck == 0u ? 1u : maxRowsToCheck);
	const unsigned int checks = (testSize < wantChecks ? testSize : wantChecks);
	const unsigned int denom = (checks > 1u ? (checks - 1u) : 1u);

	for (unsigned int k = 0; k < checks; ++k)
	{
		const unsigned int idx =
		    (testSize == 1u)
		        ? 0u
		        : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(testSize - 1u)) / denom);

		const shmea::GVector<float> row = getTestRow(idx);
		if (row.size() < expectedFeatureCount)
		{
			if (errMsg)
			{
				char buf[256];
				sprintf(buf,
				        "test row %u has %u features but expected at least %u",
				        idx,
				        static_cast<unsigned int>(row.size()),
				        expectedFeatureCount);
				*errMsg = std::string(buf);
			}
			return false;
		}

		const shmea::GVector<float> exp = getTestExpectedRow(idx);
		if (exp.size() < expectedOutSize)
		{
			if (errMsg)
			{
				char buf[256];
				sprintf(buf,
				        "test expected row %u has %u outputs but expected at least %u",
				        idx,
				        static_cast<unsigned int>(exp.size()),
				        expectedOutSize);
				*errMsg = std::string(buf);
			}
			return false;
		}
	}

	return true;
}

};

#endif
