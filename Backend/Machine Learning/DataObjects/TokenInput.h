// TokenInput: minimal token-id sequence DataInput for language modeling.
//
// Format:
// - import(path) reads a text file where each line is a sequence of integer token ids.
// - Tokens are whitespace-separated.
// - Each line becomes one sequence span in DataInput.
//
// Semantics:
// - Token IDs are stored as first-class signed integers (int).
// - Feature rows are exposed as a single float containing the token id (derived view for API compatibility).
// - Expected rows are exposed as a single float containing the next-token id (derived view).
// - For each sequence, targets are the next token in-sequence.
//   - If padTokenId >= 0: the final timestep's expected token is padTokenId (so LM loss can ignore it).
//   - If padTokenId < 0: the final timestep is not emitted (avoids negative expected token ids).
//
#pragma once

#include "DataInput.h"
#include <vector>
#include <string>

namespace glades {

class TokenInput : public DataInput
{
public:
	TokenInput();
	virtual ~TokenInput();

	// Optional: control pad/ignore token id written for sequence-final targets.
	// If set to a negative value, TokenInput will omit sequence-final timesteps entirely.
	void setPadTokenId(int id) { padTokenId = id; }
	int getPadTokenId() const { return padTokenId; }

	// DataInput API
	virtual void import(shmea::GString, int = 0);
	virtual void import(const shmea::GTable&, int = 0);

	virtual shmea::GVector<float> getTrainRow(unsigned int) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int) const;
	virtual shmea::GVector<float> getTestRow(unsigned int) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int) const;

	// View APIs:
	// - For compatibility with the legacy float-based DataInput contract, token ids are exposed
	//   as a single float.
	// - `outData` points to thread-local scratch storage and is valid until the next call to
	//   *any* TokenInput row-view method on the same thread.
	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;

	// Token-id accessors (first-class integers for token LMs).
	virtual bool getTrainTokenId(unsigned int index, int& outTokenId) const;
	virtual bool getTrainExpectedTokenId(unsigned int index, int& outTokenId) const;
	virtual bool getTestTokenId(unsigned int index, int& outTokenId) const;
	virtual bool getTestExpectedTokenId(unsigned int index, int& outTokenId) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	// Fixed shapes: featureCount=1, expectedCount=1.
	virtual bool hasFixedTrainRowSize() const { return true; }
	virtual unsigned int getFixedTrainRowSize() const { return 1u; }
	virtual bool hasFixedTrainExpectedRowSize() const { return true; }
	virtual unsigned int getFixedTrainExpectedRowSize() const { return 1u; }

	virtual int getType() const { return TEXT; }

	// Status reporting / compatibility helpers.
	bool loadedOk() const { return loaded; }
	virtual NNetworkStatus getLastStatus() const { return lastImportStatus; }

private:
	bool loaded;
	int padTokenId;

	// Most recent import/load status. When loaded==false, this SHOULD describe why.
	NNetworkStatus lastImportStatus;

	// Packed contiguous storage (row-major over timesteps), as token-id ints.
	std::vector<int> trainTok;
	std::vector<int> trainNextTok;
	std::vector<int> testTok;
	std::vector<int> testNextTok;

	// Internal helper to parse a token file.
	NNetworkStatus loadTokenFile(const std::string& path,
	                             std::vector<int>& outTok,
	                             std::vector<int>& outNext,
	                             std::vector<SequenceSpan>& outSeq,
	                             unsigned int* outLineCount = NULL);
};

} // namespace glades

