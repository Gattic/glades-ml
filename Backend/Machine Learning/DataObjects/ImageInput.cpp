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
#include "ImageInput.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "../GMath/OHE.h"
#include <algorithm>
#include <string>

using namespace glades;

namespace {
static inline std::string to_std_string(const shmea::GString& s)
{
	return std::string(s.c_str());
}

static inline shmea::GString to_gstring(const std::string& s)
{
	return shmea::GString(s.c_str());
}

// Normalize the legend's label column to a string, matching the logic in importHelper().
// This is used for test legends, where we want consistent encoding without expanding
// the label space beyond what training saw.
static inline void normalizeLegendLabelColumnToString(shmea::GTable& table, unsigned int labelCol)
{
	if (table.numberOfRows() == 0 || table.numberOfCols() == 0)
		return;
	if (labelCol >= table.numberOfCols())
		return;

	for (unsigned int r = 0; r < table.numberOfRows(); ++r)
	{
		shmea::GString label = "";
		const shmea::GType& cCell = table.getCell(r, labelCol);
		const shmea::GType::Type cType = cCell.getType();
		if (cType == shmea::GType::STRING_TYPE)
		{
			// Already normalized.
			continue;
		}
		else if (cType == shmea::GType::CHAR_TYPE)
			label = shmea::GString::intTOstring(cCell.getChar());
		else if (cType == shmea::GType::SHORT_TYPE)
			label = shmea::GString::intTOstring(cCell.getShort());
		else if (cType == shmea::GType::INT_TYPE)
			label = shmea::GString::intTOstring(cCell.getInt());
		else if (cType == shmea::GType::LONG_TYPE)
			label = shmea::GString::longTOstring(cCell.getLong());
		else if (cType == shmea::GType::BOOLEAN_TYPE)
			label = cCell.getBoolean() ? "true" : "false";
		else
		{
			// FLOAT/DOUBLE and others are not valid for classification labels here.
			// Leave as-is.
			continue;
		}

		table.setCell(r, labelCol, label);
	}
}
} // namespace

void ImageInput::importHelper(shmea::GTable& cTable, std::vector<shmea::GPointer<OHE> >& OHEMaps, std::vector<bool>& featureIsCategorical)
{
    if(loaded)
	return;

    if ((cTable.numberOfRows() == 0) || (cTable.numberOfCols() == 0))
    {
	printf("[NNDATA] Could not load data\n");
	return;
    }

    // NOTE: In streaming mode we no longer preload images into the 'images' map.
    // We only build label mappings (OHE) and ensure legend labels are normalized to string type.

    // Load the images
    unsigned int inputCol = 0;
    unsigned int outputCol = 1;
    for(unsigned int r = 0; r < cTable.numberOfRows(); ++r)
    {
	if (r == 0)
	{
	    // Really only need it for the output column for images so the first OHE will be empty
	    for(unsigned int c = 0; c < cTable.numberOfCols(); ++c)
	    {
		shmea::GPointer<OHE> cOHE(new OHE());
		featureIsCategorical.push_back(false);
		OHEMaps.push_back(cOHE);
	    }
	}

	// Convert the label to a string for classification
	shmea::GString label = "";
	const shmea::GType& cCell = cTable.getCell(r, outputCol); // get the first cell of the col
	shmea::GType::Type cType = cCell.getType();
	if (cType == shmea::GType::STRING_TYPE)
		label = cCell.c_str();
	else if (cType == shmea::GType::CHAR_TYPE)
		label = shmea::GString::intTOstring(cCell.getChar());
	else if (cType == shmea::GType::SHORT_TYPE)
		label = shmea::GString::intTOstring(cCell.getShort());
	else if (cType == shmea::GType::INT_TYPE)
		label = shmea::GString::intTOstring(cCell.getInt());
	else if (cType == shmea::GType::LONG_TYPE)
		label = shmea::GString::longTOstring(cCell.getLong());
	else if (cType == shmea::GType::FLOAT_TYPE)
	{
	    printf("Invalid type for image classification\n");
	    return;
	}
	else if (cType == shmea::GType::DOUBLE_TYPE)
	{
	    printf("Invalid type for image classification\n");
	    return;
	}
	else if (cType == shmea::GType::BOOLEAN_TYPE)
	{
		if(cCell.getBoolean())
		    label = "true";
		else
		    label = "false";
	}

	if (label.length() == 0)
	{
	    printf("Invalid type for image classification\n");
	    return;
	}

	cTable.setCell(r, outputCol, label);
	OHEMaps[outputCol]->addString(label);
	featureIsCategorical[outputCol] = true;
    }
}

void ImageInput::import(shmea::GString newName, int standardizeFlag)
{
    if(loaded)
	return;

    name = newName;
    shmea::GString fname = "datasets/images/" + name + "/";

    //
    shmea::GString trainFName = fname + "train.csv";
    shmea::GString testFName = fname + "test.csv";
    int importType = shmea::GTable::TYPE_FILE;

    trainingLegend = shmea::GTable(trainFName, ',', importType);
    testingLegend = shmea::GTable(testFName, ',', importType);

    if ((trainingLegend.numberOfRows() == 0) || (testingLegend.numberOfRows() == 0))
    {
	printf("[NNDATA] Could not load data\n");
	return;
    }

    // Build label->one-hot mappings from TRAINING ONLY.
    // IMPORTANT: output dimensionality MUST NOT change based on the test set.
    importHelper(trainingLegend, trainingOHEMaps, trainingFeatureIsCategorical);

    // Normalize test labels to string type, but do NOT add them to the OHE map.
    normalizeLegendLabelColumnToString(testingLegend, /*labelCol*/ 1u);

    // Use the training label space for test encoding.
    testingOHEMaps = trainingOHEMaps;
    testingFeatureIsCategorical = trainingFeatureIsCategorical;

    // Precompute one-hot vectors for the label space (avoid per-row allocations in hot paths).
    oneHotByIndex.clear();
    if (trainingOHEMaps.size() > 1u && trainingOHEMaps[1])
    {
        const unsigned int K = trainingOHEMaps[1]->size();
        oneHotByIndex.resize(K);
        for (unsigned int i = 0; i < K; ++i)
        {
            oneHotByIndex[i] = shmea::GVector<float>(K, 0.0f);
            oneHotByIndex[i][i] = 1.0f;
        }
    }

    // Precompute fully qualified paths for streaming access.
    trainingPaths.clear();
    testingPaths.clear();
    trainingPaths.reserve(trainingLegend.numberOfRows());
    testingPaths.reserve(testingLegend.numberOfRows());
    for (unsigned int r = 0; r < trainingLegend.numberOfRows(); ++r)
        trainingPaths.push_back(to_std_string(fname + trainingLegend.getCell(r, 0).c_str()));
    for (unsigned int r = 0; r < testingLegend.numberOfRows(); ++r)
        testingPaths.push_back(to_std_string(fname + testingLegend.getCell(r, 0).c_str()));

    // Determine feature count by loading one image (first training row).
    featureCount = 0;
    if (!trainingPaths.empty())
    {
        shmea::Image img;
        img.LoadPNG(to_gstring(trainingPaths[0]));
        featureCount = img.getPixelCount();
    }

    // Reset row cache
    rowCacheOrder.clear();
    rowCache.clear();
    scratchRow.clear();
    scratchExpected.clear();

    // Set the loaded flag
    min = 0;
    max = 255;
    loaded = true;
}

void ImageInput::import(const shmea::GTable& rawLegend, int standardizeFlag)
{
	(void)standardizeFlag;
	if (loaded)
		return;

	// Interpret the provided table as a "legend" with at least:
	// - column 0: image path (relative or absolute)
	// - column 1: classification label (string/int/bool convertible)
	//
	// This overload intentionally does NOT create a test split; callers that want a
	// train/test split should provide two tables or mirror the legend explicitly.
	trainingLegend = rawLegend;
	testingLegend = shmea::GTable(rawLegend.getDelimiter(), rawLegend.getHeaders());

	trainingOHEMaps.clear();
	testingOHEMaps.clear();
	trainingFeatureIsCategorical.clear();
	testingFeatureIsCategorical.clear();

	trainingPaths.clear();
	testingPaths.clear();
	featureCount = 0u;

	// Validate minimal legend schema.
	if (trainingLegend.numberOfCols() < 2u || trainingLegend.numberOfRows() == 0u)
	{
		printf("[NNDATA] Could not load data (legend must have >=2 cols and >=1 row)\n");
		return;
	}

	// Build label mappings from TRAINING ONLY (and normalize label column to string type).
	importHelper(trainingLegend, trainingOHEMaps, trainingFeatureIsCategorical);
	testingOHEMaps = trainingOHEMaps;
	testingFeatureIsCategorical = trainingFeatureIsCategorical;

	// Precompute one-hot vectors for the label space (avoid per-row allocations in hot paths).
	oneHotByIndex.clear();
	if (trainingOHEMaps.size() > 1u && trainingOHEMaps[1])
	{
		const unsigned int K = trainingOHEMaps[1]->size();
		oneHotByIndex.resize(K);
		for (unsigned int i = 0; i < K; ++i)
		{
			oneHotByIndex[i] = shmea::GVector<float>(K, 0.0f);
			oneHotByIndex[i][i] = 1.0f;
		}
	}

	// Precompute paths for streaming access.
	trainingPaths.reserve(trainingLegend.numberOfRows());
	for (unsigned int r = 0; r < trainingLegend.numberOfRows(); ++r)
		trainingPaths.push_back(to_std_string(trainingLegend.getCell(r, 0).c_str()));

	// Determine feature count by loading one image (first row).
	if (!trainingPaths.empty())
	{
		shmea::Image img;
		img.LoadPNG(to_gstring(trainingPaths[0]));
		featureCount = img.getPixelCount();
	}

	// Reset row cache/scratch buffers.
	rowCacheOrder.clear();
	rowCache.clear();
	scratchRow.clear();
	scratchExpected.clear();

	// Match the pixel min/max semantics (byte-range images).
	min = 0;
	max = 255;
	loaded = true;
}

const shmea::GPointer<shmea::Image> ImageInput::getTrainImage(unsigned int row) const
{
    // Streaming: load image on demand.
    if (row >= trainingPaths.size())
        return shmea::GPointer<shmea::Image>(new shmea::Image());
    shmea::GPointer<shmea::Image> img(new shmea::Image());
    img->LoadPNG(to_gstring(trainingPaths[row]));
    return img;
}

const shmea::GPointer<shmea::Image> ImageInput::getTestImage(unsigned int row) const
{
    if (row >= testingPaths.size())
        return shmea::GPointer<shmea::Image>(new shmea::Image());
    shmea::GPointer<shmea::Image> img(new shmea::Image());
    img->LoadPNG(to_gstring(testingPaths[row]));
    return img;
}

shmea::GVector<float> ImageInput::getTrainRow(unsigned int index) const
{
    if (index >= trainingPaths.size())
        return emptyRow;

    const std::string& path = trainingPaths[index];

    // LRU cache lookup
    std::map<std::string, RowCacheEntry>::iterator it = rowCache.find(path);
    if (it != rowCache.end())
    {
        // touch
        rowCacheOrder.erase(it->second.lruIt);
        rowCacheOrder.push_front(path);
        it->second.lruIt = rowCacheOrder.begin();
        return it->second.row;
    }

    // Load -> flatten -> standardize
    shmea::Image img;
    img.LoadPNG(to_gstring(path));
    shmea::GVector<float> row = img.flatten();
    row = shmea::vectorStandardize(row);

    // insert into cache
    if (rowCacheMaxEntries > 0)
    {
        if (rowCache.size() >= rowCacheMaxEntries && !rowCacheOrder.empty())
        {
            const std::string evictKey = rowCacheOrder.back();
            rowCacheOrder.pop_back();
            rowCache.erase(evictKey);
        }
        rowCacheOrder.push_front(path);
        RowCacheEntry e;
        e.row = row;
        e.lruIt = rowCacheOrder.begin();
        rowCache[path] = e;
    }

    return row;
}

shmea::GVector<float> ImageInput::getTrainExpectedRow(unsigned int index) const
{
    if(index >= trainingLegend.numberOfRows())
	return emptyRow;

    const shmea::GString& cCell = trainingLegend.getCell(index, 1);

    // translate string to cell value for this col
    const shmea::GPointer<OHE>& OHEVector = trainingOHEMaps[1];
    return (*OHEVector)[cCell];
}

shmea::GVector<float> ImageInput::getTestExpectedRow(unsigned int index) const
{
    if(index >= testingLegend.numberOfRows())
	return shmea::GVector<float>();

    const shmea::GString& cCell = testingLegend.getCell(index, 1);

    // translate string to cell value for this col
    const shmea::GPointer<OHE>& OHEVector = testingOHEMaps[1];
    return (*OHEVector)[cCell];
}

shmea::GVector<float> ImageInput::getTestRow(unsigned int index) const
{
    if (index >= testingPaths.size())
        return shmea::GVector<float>();

    const std::string& path = testingPaths[index];

    std::map<std::string, RowCacheEntry>::iterator it = rowCache.find(path);
    if (it != rowCache.end())
    {
        rowCacheOrder.erase(it->second.lruIt);
        rowCacheOrder.push_front(path);
        it->second.lruIt = rowCacheOrder.begin();
        return it->second.row;
    }

    shmea::Image img;
    img.LoadPNG(to_gstring(path));
    shmea::GVector<float> row = img.flatten();
    row = shmea::vectorStandardize(row);

    if (rowCacheMaxEntries > 0)
    {
        if (rowCache.size() >= rowCacheMaxEntries && !rowCacheOrder.empty())
        {
            const std::string evictKey = rowCacheOrder.back();
            rowCacheOrder.pop_back();
            rowCache.erase(evictKey);
        }
        rowCacheOrder.push_front(path);
        RowCacheEntry e;
        e.row = row;
        e.lruIt = rowCacheOrder.begin();
        rowCache[path] = e;
    }

    return row;
}


unsigned int ImageInput::getTrainSize() const
{
    return trainingLegend.numberOfRows();
}

unsigned int ImageInput::getTestSize() const
{
    return testingLegend.numberOfRows();
}

unsigned int ImageInput::getFeatureCount() const
{
	return featureCount;
}

int ImageInput::getType() const
{
    return IMAGE;
}

bool ImageInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
    outData = NULL;
    outSize = 0u;
    if (index >= trainingPaths.size())
        return false;

    const std::string& path = trainingPaths[index];

    std::map<std::string, RowCacheEntry>::iterator it = rowCache.find(path);
    if (it != rowCache.end())
    {
        // touch
        rowCacheOrder.erase(it->second.lruIt);
        rowCacheOrder.push_front(path);
        it->second.lruIt = rowCacheOrder.begin();
        outData = it->second.row.data();
        outSize = static_cast<unsigned int>(it->second.row.size());
        return (outData != NULL && outSize > 0u);
    }

    // Cache miss: materialize the row once.
    shmea::Image img;
    img.LoadPNG(to_gstring(path));
    shmea::GVector<float> row = img.flatten();
    row = shmea::vectorStandardize(row);

    if (rowCacheMaxEntries > 0)
    {
        if (rowCache.size() >= rowCacheMaxEntries && !rowCacheOrder.empty())
        {
            const std::string evictKey = rowCacheOrder.back();
            rowCacheOrder.pop_back();
            rowCache.erase(evictKey);
        }
        rowCacheOrder.push_front(path);
        RowCacheEntry e;
        e.row = row;
        e.lruIt = rowCacheOrder.begin();
        rowCache[path] = e;

        outData = rowCache[path].row.data();
        outSize = static_cast<unsigned int>(rowCache[path].row.size());
        return (outData != NULL && outSize > 0u);
    }

    // No caching: keep the row alive in scratch storage.
    scratchRow = row;
    outData = scratchRow.data();
    outSize = static_cast<unsigned int>(scratchRow.size());
    return (outData != NULL && outSize > 0u);
}

bool ImageInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
    outData = NULL;
    outSize = 0u;
    if (index >= testingPaths.size())
        return false;

    const std::string& path = testingPaths[index];

    std::map<std::string, RowCacheEntry>::iterator it = rowCache.find(path);
    if (it != rowCache.end())
    {
        rowCacheOrder.erase(it->second.lruIt);
        rowCacheOrder.push_front(path);
        it->second.lruIt = rowCacheOrder.begin();
        outData = it->second.row.data();
        outSize = static_cast<unsigned int>(it->second.row.size());
        return (outData != NULL && outSize > 0u);
    }

    shmea::Image img;
    img.LoadPNG(to_gstring(path));
    shmea::GVector<float> row = img.flatten();
    row = shmea::vectorStandardize(row);

    if (rowCacheMaxEntries > 0)
    {
        if (rowCache.size() >= rowCacheMaxEntries && !rowCacheOrder.empty())
        {
            const std::string evictKey = rowCacheOrder.back();
            rowCacheOrder.pop_back();
            rowCache.erase(evictKey);
        }
        rowCacheOrder.push_front(path);
        RowCacheEntry e;
        e.row = row;
        e.lruIt = rowCacheOrder.begin();
        rowCache[path] = e;

        outData = rowCache[path].row.data();
        outSize = static_cast<unsigned int>(rowCache[path].row.size());
        return (outData != NULL && outSize > 0u);
    }

    scratchRow = row;
    outData = scratchRow.data();
    outSize = static_cast<unsigned int>(scratchRow.size());
    return (outData != NULL && outSize > 0u);
}

bool ImageInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
    outData = NULL;
    outSize = 0u;
    if (index >= trainingLegend.numberOfRows())
        return false;

    // Fast path: cached one-hot vectors.
    if (!oneHotByIndex.empty() && trainingOHEMaps.size() > 1u && trainingOHEMaps[1])
    {
        const shmea::GString& label = trainingLegend.getCell(index, 1);
        const int idx = trainingOHEMaps[1]->indexAt(label);
        if (idx >= 0 && static_cast<unsigned int>(idx) < oneHotByIndex.size())
        {
            outData = oneHotByIndex[static_cast<unsigned int>(idx)].data();
            outSize = static_cast<unsigned int>(oneHotByIndex[static_cast<unsigned int>(idx)].size());
            return (outData != NULL && outSize > 0u);
        }
        // Unknown label: return an all-zeros vector of the right size.
        const unsigned int K = static_cast<unsigned int>(oneHotByIndex.size());
        if (scratchExpected.size() != K)
            scratchExpected = shmea::GVector<float>(K, 0.0f);
        outData = scratchExpected.data();
        outSize = static_cast<unsigned int>(scratchExpected.size());
        return (outData != NULL && outSize > 0u);
    }
	// No cached one-hot space available (invalid/unsupported state).
	return false;
}

bool ImageInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
    outData = NULL;
    outSize = 0u;
    if (index >= testingLegend.numberOfRows())
        return false;

    if (!oneHotByIndex.empty() && testingOHEMaps.size() > 1u && testingOHEMaps[1])
    {
        const shmea::GString& label = testingLegend.getCell(index, 1);
        const int idx = testingOHEMaps[1]->indexAt(label);
        if (idx >= 0 && static_cast<unsigned int>(idx) < oneHotByIndex.size())
        {
            outData = oneHotByIndex[static_cast<unsigned int>(idx)].data();
            outSize = static_cast<unsigned int>(oneHotByIndex[static_cast<unsigned int>(idx)].size());
            return (outData != NULL && outSize > 0u);
        }
        const unsigned int K = static_cast<unsigned int>(oneHotByIndex.size());
        if (scratchExpected.size() != K)
            scratchExpected = shmea::GVector<float>(K, 0.0f);
        outData = scratchExpected.data();
        outSize = static_cast<unsigned int>(scratchExpected.size());
        return (outData != NULL && outSize > 0u);
    }
	// No cached one-hot space available (invalid/unsupported state).
	return false;
}
