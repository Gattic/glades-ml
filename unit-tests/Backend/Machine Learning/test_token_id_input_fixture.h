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

#ifndef _UT_TEST_TOKEN_ID_INPUT_FIXTURE
#define _UT_TEST_TOKEN_ID_INPUT_FIXTURE

#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"

#include <vector>

#if __cplusplus >= 201103L
#define UT_GLADES_THREAD_LOCAL thread_local
#elif defined(_MSC_VER)
#define UT_GLADES_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define UT_GLADES_THREAD_LOCAL __thread
#else
#define UT_GLADES_THREAD_LOCAL
#endif

class InMemoryTokenIdInput : public glades::DataInput
{
public:
	InMemoryTokenIdInput()
	    : padTokenId_(-1),
	      one_(1, 0.0f),
	      empty_()
	{
	}

	void setTrainTokens(const std::vector<unsigned int>& toks, int pad)
	{
		padTokenId_ = pad;
		trainTok_.clear();
		trainNextTok_.clear();
		trainTok_.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
			trainTok_.push_back(static_cast<int>(toks[i]));
		buildNext_(trainTok_, padTokenId_, trainNextTok_);
	}

	void setTestTokens(const std::vector<unsigned int>& toks, int pad)
	{
		padTokenId_ = pad;
		testTok_.clear();
		testNextTok_.clear();
		testTok_.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
			testTok_.push_back(static_cast<int>(toks[i]));
		buildNext_(testTok_, padTokenId_, testNextTok_);
	}

	void mirrorTrainToTest()
	{
		testTok_ = trainTok_;
		testNextTok_ = trainNextTok_;
	}

	virtual void import(shmea::GString, int = 0) {}
	virtual void import(const shmea::GTable&, int = 0) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int i) const
	{
		if (i >= trainTok_.size())
			return empty_;
		one_[0] = static_cast<float>(trainTok_[i]);
		return one_;
	}

	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{
		if (i >= trainNextTok_.size())
			return empty_;
		one_[0] = static_cast<float>(trainNextTok_[i]);
		return one_;
	}

	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{
		if (i >= testTok_.size())
			return empty_;
		one_[0] = static_cast<float>(testTok_[i]);
		return one_;
	}

	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{
		if (i >= testNextTok_.size())
			return empty_;
		one_[0] = static_cast<float>(testNextTok_[i]);
		return one_;
	}

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainTok_.size())
			return false;
		static UT_GLADES_THREAD_LOCAL float tlsTok;
		tlsTok = static_cast<float>(trainTok_[index]);
		outData = &tlsTok;
		outSize = 1u;
		return true;
	}

	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainNextTok_.size())
			return false;
		static UT_GLADES_THREAD_LOCAL float tlsNextTok;
		tlsNextTok = static_cast<float>(trainNextTok_[index]);
		outData = &tlsNextTok;
		outSize = 1u;
		return true;
	}

	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testTok_.size())
			return false;
		static UT_GLADES_THREAD_LOCAL float tlsTok;
		tlsTok = static_cast<float>(testTok_[index]);
		outData = &tlsTok;
		outSize = 1u;
		return true;
	}

	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testNextTok_.size())
			return false;
		static UT_GLADES_THREAD_LOCAL float tlsNextTok;
		tlsNextTok = static_cast<float>(testNextTok_[index]);
		outData = &tlsNextTok;
		outSize = 1u;
		return true;
	}

	virtual bool getTrainTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainTok_.size())
			return false;
		outTokenId = trainTok_[index];
		return true;
	}

	virtual bool getTrainExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainNextTok_.size())
			return false;
		outTokenId = trainNextTok_[index];
		return true;
	}

	virtual bool getTestTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testTok_.size())
			return false;
		outTokenId = testTok_[index];
		return true;
	}

	virtual bool getTestExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testNextTok_.size())
			return false;
		outTokenId = testNextTok_[index];
		return true;
	}

	virtual bool hasTokenIdInput() const { return true; }
	virtual bool hasTokenIdExpectedOutput() const { return true; }

	virtual unsigned int getTrainSize() const { return static_cast<unsigned int>(trainTok_.size()); }
	virtual unsigned int getTestSize() const { return static_cast<unsigned int>(testTok_.size()); }
	virtual unsigned int getFeatureCount() const { return 1u; }
	virtual int getType() const { return TEXT; }

private:
	static void buildNext_(const std::vector<int>& toks, int pad, std::vector<int>& outNext)
	{
		outNext.clear();
		outNext.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
			outNext.push_back((i + 1u < toks.size()) ? toks[i + 1u] : pad);
	}

	int padTokenId_;
	std::vector<int> trainTok_;
	std::vector<int> trainNextTok_;
	std::vector<int> testTok_;
	std::vector<int> testNextTok_;
	mutable shmea::GVector<float> one_;
	shmea::GVector<float> empty_;
};

#endif
