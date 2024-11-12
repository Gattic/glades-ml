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

using namespace glades;

void NumberInput::import(shmea::GString fname)
{
    if(loaded)
	return;

    name = fname;

    // Load and Normalize/Standardize the data
    standardizeInputTable(fname);

    // TODO: test table stuff

    // Set the loaded flag
    loaded = true;
}

void glades::NumberInput::standardizeInputTable(const shmea::GString& inputFName, int standardizeFlag)
{
	trainTable = shmea::GTable(',');
	shmea::GTable rawTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);

	// Standardize the initialization of the weights
	if ((rawTable.numberOfRows() <= 0) || (rawTable.numberOfCols() <= 0))
		return;

	// default cols to non-categorical
	bool isClassification = false;
	for (unsigned int c = 0; c < rawTable.numberOfCols(); ++c)
	{
		OHE* cOHE = new OHE();
		featureIsCategorical.push_back(false);

		shmea::GType cCell = rawTable.getCell(0, c); // get the first cell of the col
		if (cCell.getType() == shmea::GType::STRING_TYPE)
		{
			cOHE->mapFeatureSpace(rawTable, c);
			featureIsCategorical[c] = true;
			isClassification = true;
			cOHE->print();
		}

		OHEMaps.push_back(cOHE);
	}

	// iterate through the cols
	for (unsigned int c = 0; c < rawTable.numberOfCols(); ++c)
	{
		// Set the min and max for this feature (col)
		float fMin = 0.0f;
		float fMax = 0.0f;
		float fMean = 0.0f;

		// iterate through the rows
		for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r)
		{
			// check if already marked categorical
			if (featureIsCategorical[c])
				continue;

			float cell = 0.0f;
			shmea::GType cCell = rawTable.getCell(r, c);

			if (cCell.getType() == shmea::GType::STRING_TYPE)
				continue; // mapped strings in a previous loop
			else if (cCell.getType() == shmea::GType::CHAR_TYPE)
				cell = cCell.getChar();
			else if (cCell.getType() == shmea::GType::SHORT_TYPE)
				cell = cCell.getShort();
			else if (cCell.getType() == shmea::GType::INT_TYPE)
				cell = cCell.getInt();
			else if (cCell.getType() == shmea::GType::LONG_TYPE)
				cell = cCell.getLong();
			else if (cCell.getType() == shmea::GType::FLOAT_TYPE)
				cell = cCell.getFloat();
			else if (cCell.getType() == shmea::GType::DOUBLE_TYPE)
				cell = cCell.getDouble();
			else if (cCell.getType() == shmea::GType::BOOLEAN_TYPE)
				cell = cCell.getBoolean() ? 1.0f : 0.0f;

			if ((r == 0) && (c == 0))
			{
				fMin = cell;
				fMax = cell;
			}

			// Check the mins and maxes
			if (cell < fMin)
				fMin = cell;
			if (cell > fMax)
				fMax = cell;

			// update mean
			fMean += cell;
			if (r == (rawTable.numberOfRows() - 1))
				fMean /= rawTable.numberOfRows();
		}
		printf("c: %d:%u, fMin: %f, fMax: %f, fMean: %f\n", c, rawTable.numberOfCols(), fMin, fMax, fMean);

		if (featureIsCategorical[c])
		{
			OHE* OHEVector = OHEMaps[c];
			printf("OHEVector size: %d\n", OHEVector->size());

			// iterate over feature (col) space
			for (unsigned int cInt = 0; cInt < OHEVector->size(); ++cInt)
			{
				// iterate over rows
				shmea::GList newCol;
				for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r)
				{
					shmea::GType cCell = rawTable.getCell(r, c);
					float cell = 0.0f;

					// translate string to cell value for this col
					std::string cString = cCell.c_str();
					std::vector<float> featureVector = (*OHEVector)[cString];
					cell = featureVector[cInt];

					// add cell to newCol
					newCol.addFloat(cell);
				}

				// generic new header since OHE turns 1 col to many
				shmea::GString newHeader = rawTable.getHeader(c).c_str();
				newHeader += shmea::GString::intTOstring(cInt);

				// add the standardized newCol to the trainTable
				if (rawTable.isOutput(c))
				{
				    trainExpectedTable.addCol(newHeader, newCol);
				    trainExpectedTable.toggleOutput(trainExpectedTable.numberOfCols() - 1);
				}
				else
				    trainTable.addCol(newHeader, newCol);
			}
		}
		else
		{
			if (standardizeFlag == GMath::MINMAX)
			{
				// find the range of this feature
				float xRange = fMax - fMin;
				if (xRange == 0.0f)
				{
					// This column is just a constant, add and skip standardization
					trainTable.addCol(rawTable.getHeader(c), rawTable.getCol(c));
					continue;
				}

				// iterate through the rows
				shmea::GList newCol;
				for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r)
				{
					// acquire original cell value
					shmea::GType cCell = rawTable.getCell(r, c);
					float cell = 0.0f;
					if (cCell.getType() == shmea::GType::STRING_TYPE)
					{
						// for errors - strings MUST be categorical
						printf("ERROR: String found in non-categorical column.\n");
						trainTable.clear();
						return;
					}
					else
						cell = cCell.getFloat();

					// standardize cell value based on network vars
					if (isClassification) // CLASSIFICATION
					{
						// [0.01, 0.99] bounds
						cell = ((((cell - fMin) / (xRange)) * 0.98f) + 0.01f);
					}
					else // REGRESSION
					{
						/*int activationType = skeleton->getActivationType(0); // 0 because first layer
						if ((activationType == GMath::SIGMOID) || (activationType ==
						GMath::SIGMOIDP)
						||
							(activationType == GMath::RELU) || (activationType == GMath::LEAKY))
							cell = ((cell - fMin) / (xRange)); // [0.0, 1.0] bounds
						else
							cell = ((((cell - fMin) / (xRange)) * 2.0f) - 1.0f); // [-1.0, 1.0]
						bounds*/

						// [0.0, 1.0] bounds
						cell = ((cell - fMin) / (xRange));
					}

					// add cell to newCol
					newCol.addFloat(cell);
				}

				// add the standardized newCol to the trainTable
				if (rawTable.isOutput(c))
				{
					trainExpectedTable.addCol(rawTable.getHeader(c), newCol);
					trainExpectedTable.toggleOutput(trainTable.numberOfCols() - 1);
				}
				else
				    trainTable.addCol(rawTable.getHeader(c), newCol);
			}
			else if (standardizeFlag == GMath::ZSCORE)
			{
				// second pass for stdev
				float fStDev = 0.0f;
				shmea::GList newCol;
				for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r)
				{
					// acquire original cell value
					shmea::GType cCell = rawTable.getCell(r, c);
					float cell = 0.0f;
					if (cCell.getType() == shmea::GType::STRING_TYPE)
					{
						// for errors - strings MUST be categorical
						trainTable.clear();
						return;
					}
					else
						cell = cCell.getFloat();

					fStDev += ((cell - fMean) * (cell - fMean));
				}

				// calculate stdev
				fStDev = sqrt(fStDev / (rawTable.numberOfRows() - 1));

				for (unsigned int r = 0; r < rawTable.numberOfRows(); ++r)
				{
					// acquire original cell value
					shmea::GType cCell = rawTable.getCell(r, c);
					float cell = 0.0f;
					if (cCell.getType() == shmea::GType::STRING_TYPE)
					{
						// for errors - strings MUST be categorical
						printf("ERROR: String found in non-categorical column.\n");
						trainTable.clear();
						return;
					}
					else
						cell = cCell.getFloat();

					cell = ((cell - fMean) / fStDev);

					// add cell to newCol
					newCol.addFloat(cell);
				}

				// add the standardized newCol to the trainTable
				if (rawTable.isOutput(c))
				{
					trainExpectedTable.addCol(rawTable.getHeader(c), newCol);
					trainExpectedTable.toggleOutput(trainTable.numberOfCols() - 1);
				}
				else
				    trainTable.addCol(rawTable.getHeader(c), newCol);
			}
		}
	}
}

shmea::GList NumberInput::getTrainRow(unsigned int index) const
{
    if(index >= trainTable.numberOfRows())
	return emptyRow;

    return trainTable.getRow(index);
}

shmea::GList NumberInput::getTrainExpectedRow(unsigned int index) const
{
    if(index >= trainExpectedTable.numberOfRows())
	return emptyRow;

    return trainExpectedTable.getRow(index);
}

shmea::GList NumberInput::getTestRow(unsigned int index) const
{
    if(index >= testTable.numberOfRows())
	return shmea::GList();

    return testTable.getRow(index);
}

shmea::GList NumberInput::getTestExpectedRow(unsigned int index) const
{
    if(index >= testExpectedTable.numberOfRows())
	return shmea::GList();

    return testExpectedTable.getRow(index);
}

unsigned int NumberInput::getTrainSize() const
{
    return trainTable.numberOfRows();
}

unsigned int NumberInput::getTestSize() const
{
    return testTable.numberOfRows();
}

unsigned int NumberInput::getFeatureCount() const
{
    return trainTable[0].size();
}

int NumberInput::getType() const
{
    return CSV;
}
