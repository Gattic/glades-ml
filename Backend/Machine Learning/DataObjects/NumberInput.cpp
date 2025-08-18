// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
    standardizeInputTable(rawTable, standardizeFlag, false);

    loaded = true;
}

//void glades::NumberInput::standardizeInputTable(const shmea::GString& inputFName, int standardizeFlag, bool changeValues)
void glades::NumberInput::standardizeInputTable(const shmea::GTable& rawTable, int standardizeFlag, bool changeValues)
{
//    shmea::GTable rawTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);

    // Standardize the initialization of the weights
    if ((rawTable.numberOfRows() <= 0) || (rawTable.numberOfCols() <= 0))
        return;

    // Initialize matrices with proper dimensions
    unsigned int inputColIdx = 0;
    unsigned int outputColIdx = 0;
    unsigned int totalInputCols = 0;
    unsigned int totalOutputCols = 0;

    // default cols to non-categorical
    bool isClassification = false;
    for (unsigned int c = 0; c < rawTable.numberOfCols(); ++c)
    {
        OHE* cOHE = new OHE();
        trainingFeatureIsCategorical.push_back(false);

        shmea::GType cCell = rawTable.getCell(0, c); // get the first cell of the col
        if (cCell.getType() == shmea::GType::STRING_TYPE)
        {
            cOHE->mapFeatureSpace(rawTable, c);
            trainingFeatureIsCategorical[c] = true;
            isClassification = true;
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
            OHE* OHEVector = trainingOHEMaps[c];
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
            float fMean = 0.0f;

            // First get min/max/mean
            for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r) 
            {
                float cell = rawTable.getCell(r, c).getFloat();
                if (r == 0) 
                {
                    fMin = cell;
                    fMax = cell;
                }
                if (cell < fMin) fMin = cell;
                if (cell > fMax) fMax = cell;
                fMean += cell;
            }
            fMean /= rawTable.numberOfRows();

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
                        // Calculate standard deviation
                        float fStDev = 0.0f;
                        for (unsigned int i = 0; i < rawTable.numberOfRows(); ++i) 
                        {
                            float val = rawTable.getCell(i, c).getFloat();
                            fStDev += ((val - fMean) * (val - fMean));
                        }
                        fStDev = sqrt(fStDev / (rawTable.numberOfRows() - 1));
                        if (fStDev != 0.0f) 
                        {
                            cell = ((cell - fMean) / fStDev);
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
    return trainMatrix[0].size();
}

int NumberInput::getType() const
{
    return CSV;
}
