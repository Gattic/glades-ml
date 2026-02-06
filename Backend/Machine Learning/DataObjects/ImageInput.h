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
#ifndef _GIMAGEINPUT
#define _GIMAGEINPUT

#include "DataInput.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/image.h"
#include <stdio.h>
#include <list>
#include <vector>
#include <map>
#include <string>

namespace glades {

class ImageInput : public DataInput
{
public:

	// Path, Label
	shmea::GTable trainingLegend;
	shmea::GTable testingLegend;

	shmea::GVector<float> emptyRow;
	shmea::GString name;
	bool loaded;

	// Streaming metadata (precomputed at import time).
	std::vector<std::string> trainingPaths; // fully-qualified paths
	std::vector<std::string> testingPaths;  // fully-qualified paths
	unsigned int featureCount;

	// Row cache (LRU) for flattened+standardized image tensors.
	// Keyed by full path. Mutable so getTrainRow/getTestRow can cache under const API.
	struct RowCacheEntry
	{
		shmea::GVector<float> row;
		std::list<std::string>::iterator lruIt;
	};

	mutable std::list<std::string> rowCacheOrder; // front = most recently used
	mutable std::map<std::string, RowCacheEntry> rowCache;
	unsigned int rowCacheMaxEntries;

	// Scratch storage for view APIs when caching is disabled or a miss occurs.
	mutable shmea::GVector<float> scratchRow;
	mutable shmea::GVector<float> scratchExpected;

	// Cached one-hot vectors for labels (avoids per-timestep allocations in hot paths).
	// Index corresponds to trainingOHEMaps[1]->indexAt(label).
	std::vector< shmea::GVector<float> > oneHotByIndex;

	ImageInput()
	{
		//
	    name = "";
	    loaded = false;
	    trainingLegend.clear();
	    testingLegend.clear();
	    trainingPaths.clear();
	    testingPaths.clear();
	    featureCount = 0;
	    rowCacheOrder.clear();
	    rowCache.clear();
	    rowCacheMaxEntries = 64;
	}

	virtual ~ImageInput()
	{
	    name = "";
	    loaded = false;
	    trainingLegend.clear();
	    testingLegend.clear();
	    trainingPaths.clear();
	    testingPaths.clear();
	    featureCount = 0;
	    rowCacheOrder.clear();
	    rowCache.clear();
	}

	void importHelper(shmea::GTable&, std::vector<shmea::GPointer<OHE> >&, std::vector<bool>&);

	virtual void import(shmea::GString, int = 0);
	virtual void import(const shmea::GTable&, int = 0);
	const shmea::GPointer<shmea::Image> getTrainImage(unsigned int) const;
	const shmea::GPointer<shmea::Image> getTestImage(unsigned int) const;

	virtual shmea::GVector<float> getTrainRow(unsigned int) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int) const;

	virtual shmea::GVector<float> getTestRow(unsigned int) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int) const;

	// Zero-copy row/expected row views.
	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	// Fast shape contract (prevents TrainingCore from materializing rows to validate sizes).
	virtual bool hasFixedTrainRowSize() const { return featureCount > 0u; }
	virtual unsigned int getFixedTrainRowSize() const { return featureCount; }
	virtual bool hasFixedTrainExpectedRowSize() const
	{
		return (trainingOHEMaps.size() > 1u) && (trainingOHEMaps[1]) && (trainingOHEMaps[1]->size() > 0u);
	}
	virtual unsigned int getFixedTrainExpectedRowSize() const
	{
		if (trainingOHEMaps.size() <= 1u || !trainingOHEMaps[1])
			return 0u;
		return static_cast<unsigned int>(trainingOHEMaps[1]->size());
	}

	virtual int getType() const;
};
};

#endif
