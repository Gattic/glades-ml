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
#include "NumberInput.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "../GMath/OHE.h"
#include "../GMath/gmath.h"
#include "../Structure/nninfo.h"
#include <vector>
#include <limits>

using namespace glades;

void NumberInput::import(shmea::GString fname, int standardizeFlag)
{
    if(loaded)
    {
        return;
    }

    name = fname;

    // Load and Normalize/Standardize the data
    shmea::GTable rawTable = shmea::GTable(fname, ',', shmea::GTable::TYPE_FILE);
//    standardizeInputTable(fname);
    standardizeInputTable(rawTable, standardizeFlag);

    // TODO: test table stuff

    // Set the loaded flag
    loaded = true;
}

void NumberInput::import(const shmea::GTable& rawTable, int standardizeFlag)
{
    if(loaded)
    {
        return;
    }

    // Load and Normalize/Standardize the data
    const bool changeValues = (standardizeFlag != GMath::NONE);
    standardizeInputTable(rawTable, standardizeFlag, changeValues);

    loaded = true;
}

//void glades::NumberInput::standardizeInputTable(const shmea::GString& inputFName, int standardizeFlag, bool changeValues)
void glades::NumberInput::standardizeInputTable(const shmea::GTable& rawTable, int standardizeFlag, bool changeValues)
{
//    shmea::GTable rawTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);

    // Standardize the initialization of the weights
    if ((rawTable.numberOfRows() <= 0) || (rawTable.numberOfCols() <= 0))
        return;

    // Reset global min/max tracking for this import.
    // DataInput initializes these to sentinel extremes, but NumberInput may be reused or
    // may contain only categorical columns (in which case we must not leave infinities).
    const float initMin = std::numeric_limits<float>::max();
    const float initMax = -std::numeric_limits<float>::max();
    min = initMin;
    max = initMax;
    bool sawNumeric = false;

    // Initialize matrices with proper dimensions
    unsigned int inputColIdx = 0;
    unsigned int outputColIdx = 0;
    unsigned int totalInputCols = 0;
    unsigned int totalOutputCols = 0;

    // default cols to non-categorical
    for (unsigned int c = 0; c < rawTable.numberOfCols(); ++c)
    {
        shmea::GPointer<OHE> cOHE(new OHE());
        trainingFeatureIsCategorical.push_back(false);

        shmea::GType cCell = rawTable.getCell(0, c); // get the first cell of the col
        if (cCell.getType() == shmea::GType::STRING_TYPE)
        {
            cOHE->mapFeatureSpace(rawTable, c);
            trainingFeatureIsCategorical[c] = true;
            cOHE->print();
        }

        trainingOHEMaps.push_back(cOHE);

	// Count dimensions
        if (rawTable.isOutput(c)) 
        {
            if (trainingFeatureIsCategorical[c]) 
            {
                totalOutputCols += trainingOHEMaps[c]->size();
            } 
            else 
            {
                totalOutputCols++;
            }
        } 
        else 
        {
            if (trainingFeatureIsCategorical[c]) 
            {
                totalInputCols += trainingOHEMaps[c]->size();
            } 
            else 
            {
                totalInputCols++;
            }
        }
    }

    // Initialize matrices
    trainMatrix = shmea::GMatrix(rawTable.numberOfRows(), shmea::GVector<float>(totalInputCols, 0.0f));
    trainExpectedMatrix = shmea::GMatrix(rawTable.numberOfRows(), shmea::GVector<float>(totalOutputCols, 0.0f));
    
    // Second pass: populate matrices
    for (unsigned int c = 0; c < rawTable.numberOfCols(); c++) 
    {
        if (trainingFeatureIsCategorical[c]) 
        {
            const shmea::GPointer<OHE>& OHEVector = trainingOHEMaps[c];
            for (unsigned int cInt = 0; cInt < OHEVector->size(); ++cInt) 
            {
                for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r) 
                {
                    shmea::GType cCell = rawTable.getCell(r, c);
                    shmea::GString cString = cCell.c_str();
                    shmea::GVector<float> featureVector = (*OHEVector)[cString];
                    float cell = featureVector[cInt];

                    if (rawTable.isOutput(c)) 
                    {
                        trainExpectedMatrix[r][outputColIdx] = cell;
                    } 
                    else 
                    {
                        trainMatrix[r][inputColIdx] = cell;
                    }
                }
                if (rawTable.isOutput(c)) 
                {
                    outputColIdx++;
                } 
                else 
                {
                    inputColIdx++;
                }
            }
        } 
        else 
        {
            // Handle numeric columns
            float fMin = 0.0f;
            float fMax = 0.0f;
            sawNumeric = true;
            // Use stable one-pass stats for ZSCORE.
            // (Welford) gives mean and unbiased variance in O(n).
            double mean = 0.0;
            double m2 = 0.0;
            unsigned int count = 0;

            // First get min/max/mean/(m2)
            for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r) 
            {
                const float cell = rawTable.getCell(r, c).getFloat();
                if (r == 0) 
                {
                    fMin = cell;
                    fMax = cell;
                }
                if (cell < fMin) fMin = cell;
                if (cell > fMax) fMax = cell;

                // Running mean/variance (unconditionally; it's cheap and avoids branching).
                ++count;
                const double x = static_cast<double>(cell);
                const double delta = x - mean;
                mean += delta / static_cast<double>(count);
                const double delta2 = x - mean;
                m2 += delta * delta2;
            }

            // Compute stdev once per column (unbiased estimator: divide by n-1).
            float fStDev = 0.0f;
            if (count > 1)
            {
                const double variance = m2 / static_cast<double>(count - 1);
                if (variance > 0.0)
                    fStDev = static_cast<float>(sqrt(variance));
            }

	    // Set the class min/max
	    if (fMin < min)
		min = fMin;
	    if (fMax > max)
		max = fMax;

            // Then standardize and store
            float xRange = fMax - fMin;
            for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r) 
            {
                float cell = rawTable.getCell(r, c).getFloat();
                
                if (xRange != 0.0f && changeValues) 
                {
                    if (standardizeFlag == GMath::MINMAX) 
                    {
                        cell = ((cell - fMin) / xRange);
                    } 
                    else if (standardizeFlag == GMath::ZSCORE) 
                    {
                        if (fStDev != 0.0f) 
                        {
                            cell = static_cast<float>((static_cast<double>(cell) - mean) / static_cast<double>(fStDev));
                        }
                    }
                }

                if (rawTable.isOutput(c)) 
                {
                    trainExpectedMatrix[r][outputColIdx] = cell;
                } 
                else 
                {
                    trainMatrix[r][inputColIdx] = cell;
                }
            }
            if (rawTable.isOutput(c)) 
            {
                outputColIdx++;
            } 
            else 
            {
                inputColIdx++;
            }
        }
    }

    // If there were no numeric columns at all, min/max were never updated from sentinels.
    // Keep this well-defined for downstream consumers.
    if (!sawNumeric)
    {
        min = 0.0f;
        max = 0.0f;
    }
}

shmea::GVector<float> NumberInput::getTrainRow(unsigned int index) const
{
    if(index >= trainMatrix.size())
    {
        return emptyRow;
    }
    return trainMatrix[index];
}

shmea::GVector<float> NumberInput::getTrainExpectedRow(unsigned int index) const
{
    if(index >= trainExpectedMatrix.size())
    {
        return emptyRow;
    }
    return trainExpectedMatrix[index];
}

shmea::GVector<float> NumberInput::getTestRow(unsigned int index) const
{
    if(index >= testMatrix.size())
    {
        return shmea::GVector<float>();
    }
    return testMatrix[index];
}

shmea::GVector<float> NumberInput::getTestExpectedRow(unsigned int index) const
{
    if(index >= testExpectedMatrix.size())
    {
        return shmea::GVector<float>();
    }
    return testExpectedMatrix[index];
}

bool NumberInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= trainMatrix.size())
		return false;
	const shmea::GVector<float>& row = trainMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

bool NumberInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= trainExpectedMatrix.size())
		return false;
	const shmea::GVector<float>& row = trainExpectedMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

bool NumberInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= testMatrix.size())
		return false;
	const shmea::GVector<float>& row = testMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

bool NumberInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= testExpectedMatrix.size())
		return false;
	const shmea::GVector<float>& row = testExpectedMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

unsigned int NumberInput::getTrainSize() const
{
    return trainMatrix.size();
}

unsigned int NumberInput::getTestSize() const
{
    return testMatrix.size();
}

unsigned int NumberInput::getFeatureCount() const
{
    if (trainMatrix.size() == 0)
        return 0;
    return trainMatrix[0].size();
}

int NumberInput::getType() const
{
    return CSV;
}
