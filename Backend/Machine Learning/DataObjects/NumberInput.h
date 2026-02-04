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
#ifndef _GNUMBERINPUT
#define _GNUMBERINPUT

#include "DataInput.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/image.h"
#include <stdio.h>
#include <vector>
#include <map>

namespace glades 
{

class NumberInput : public DataInput
{
public:

	// Path, Label
	shmea::GMatrix trainMatrix;
	shmea::GMatrix trainExpectedMatrix;
	shmea::GMatrix testMatrix;
	shmea::GMatrix testExpectedMatrix;

	shmea::GVector<float> emptyRow;
	shmea::GString name;
	bool loaded;

	NumberInput() :
	    name(""),
	    loaded(false)
	{
	    trainingOHEMaps.clear();
	    testingOHEMaps.clear();
	    trainingFeatureIsCategorical.clear();
	    testingFeatureIsCategorical.clear();
	    trainMatrix.clear();
	    trainExpectedMatrix.clear();
	    testMatrix.clear();
	    testExpectedMatrix.clear();
	}

	virtual ~NumberInput()
	{
	    name = "";
	    loaded = false;
	    trainingOHEMaps.clear();
	    testingOHEMaps.clear();
	    trainingFeatureIsCategorical.clear();
	    testingFeatureIsCategorical.clear();
	    trainMatrix.clear();
	    trainExpectedMatrix.clear();
	    testMatrix.clear();
	    testExpectedMatrix.clear();
	}

	virtual void import(shmea::GString, int = 0);
	virtual void import(const shmea::GTable&, int = 0);
//	void standardizeInputTable(const shmea::GString&, int = 0, bool = true);
	void standardizeInputTable(const shmea::GTable&, int = 0, bool = true);

	virtual shmea::GVector<float> getTrainRow(unsigned int) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int) const;

	virtual shmea::GVector<float> getTestRow(unsigned int) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int) const;

	// Zero-copy views into the underlying matrices.
	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	// Fast shape contract (prevents TrainingCore from materializing rows to validate sizes).
	virtual bool hasFixedTrainRowSize() const { return (trainMatrix.size() > 0) && (trainMatrix[0].size() > 0); }
	virtual unsigned int getFixedTrainRowSize() const { return getFeatureCount(); }
	virtual bool hasFixedTrainExpectedRowSize() const { return (trainExpectedMatrix.size() > 0) && (trainExpectedMatrix[0].size() > 0); }
	virtual unsigned int getFixedTrainExpectedRowSize() const
	{
		if (trainExpectedMatrix.size() == 0)
			return 0u;
		return static_cast<unsigned int>(trainExpectedMatrix[0].size());
	}

	virtual int getType() const;
};

};

#endif
