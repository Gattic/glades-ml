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
#include "bayes.h"
#include "../GMath/OHE.h"
#include "../GMath/gmath.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"

using namespace glades;

shmea::GTable NaiveBayes::import2(const shmea::GTable& newInputTable)
{
	shmea::GTable standardizedTable(',');
	standardizedTable.setHeaders(newInputTable.getHeaders());

	// Standardize the initialization of the weights
	if ((newInputTable.numberOfRows() <= 0) || (newInputTable.numberOfCols() <= 0))
		return standardizedTable;

	// iterate through the cols
	for (unsigned int c = 0; c < newInputTable.numberOfCols(); ++c)
	{
		// Set the min and max for this feature (col)
		OHE cOHE;
		cOHE.mapFeatureSpace(newInputTable, c);
		OHEMaps.push_back(cOHE);
	}

	// Convert to classMap table
	for (unsigned int r = 0; r < newInputTable.numberOfRows(); ++r)
	{
		shmea::GList newRow;
		for (unsigned int c = 0; c < newInputTable.numberOfCols(); ++c)
		{
			const OHE& cOHE = OHEMaps[c];
			float cell = 0.0f;
			shmea::GType cCell = newInputTable.getCell(r, c);
			if (cCell.getType() == shmea::GType::STRING_TYPE)
			{
			//
				shmea::GString cString = cCell;
				int featureInt = cOHE.indexAt(cString);
				newRow.addInt(featureInt);
			}
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

			newRow.addFloat(cOHE.standardize(cell));
		}

		standardizedTable.addRow(newRow);
	}

	return standardizedTable;
}


shmea::GTable glades::NaiveBayes::import(const shmea::GTable& newInputTable)
{
	shmea::GTable standardizedTable(',');
	standardizedTable.setHeaders(newInputTable.getHeaders());

	if ((newInputTable.numberOfRows() <= 0) || (newInputTable.numberOfCols() <= 0))
		return standardizedTable;

	// iterate through the cols
	for (unsigned int c = 0; c < newInputTable.numberOfCols(); ++c)
	{
		// iterate through the rows
		OHE cOHE;
		cOHE.mapFeatureSpace(newInputTable, c);
		OHEMaps.push_back(cOHE);
	}

	// Convert to classMap table
	for (unsigned int r = 0; r < newInputTable.numberOfRows(); ++r)
	{
		shmea::GList newRow;

		for (unsigned int c = 0; c < newInputTable.numberOfCols(); ++c)
		{
			shmea::GType cCell = newInputTable.getCell(r, c);
			if (cCell.getType() != shmea::GType::STRING_TYPE)
				continue;

			//
			OHE cOHE = OHEMaps[c];
			shmea::GString cString = cCell;
			int featureInt = cOHE.indexAt(cString);
			newRow.addInt(featureInt);
		}
		standardizedTable.addRow(newRow);
	}

	return standardizedTable;
}

void NaiveBayes::train(const shmea::GTable& data)
{
	int outCol = data.numberOfCols()-1;

	// count all classes and attributes
	for(unsigned int i =0; i < data.numberOfRows(); ++i)
	{
		shmea::GList row = data[i];
		if(classes.find(row[outCol]) == classes.end())
		{
			classes[row[outCol]]=1;
			std::map<int, double> pxc;
			attributesPerClass[row[outCol]] = pxc;
		}
		else
		{
			classes[row[outCol]]+=1;
		}

		for(unsigned int k=0; k <= data.numberOfCols()-1; ++k)
		{
			if(attributesPerClass[row[outCol]].find(row[k]) == attributesPerClass[row[outCol]].end())
			{
				attributesPerClass[row[outCol]][row[k]] = 1;
			}
			else
			{
				attributesPerClass[row[outCol]][row[k]] += 1;
			}
		}
	}

	std::map<int, std::map<int, double> >::iterator itr = attributesPerClass.begin();
	for (; itr != attributesPerClass.end(); ++itr)
	{
		std::map<int, double>::iterator itr2 = itr->second.begin();
		for (; itr2 != itr->second.end(); ++itr2)
		{
			itr2->second /= classes[itr->first]; // normalization
		}

		classes[itr->first] /= data.numberOfRows(); // normalization
	}
}

int NaiveBayes::predict(const shmea::GList& attributes)
{
	int outCol = OHEMaps.size()-1;

	int maxcid = -1;
	double maxp = 0;
	std::map<int, double>::iterator itr = classes.begin();
	for (; itr != classes.end(); ++itr)
	{
		// p(C|x) = p(C)*p(x1|C)*p(x2|C)*etc
		double pCx = itr->second;
		for(unsigned int i = 0; i<attributes.size(); ++i)
			pCx *= attributesPerClass[itr->first][OHEMaps[i].indexAt(attributes[i])];

		// Select the highest probability
		if(pCx > maxp)
		{
			maxp = pCx;
			maxcid = itr->first;
		}
	}

	//printf("Predicted Class: %d(%s) P(C|x) =%f\n", maxcid, OHEMaps[outCol].classAt(maxcid).c_str(), maxp);
	return maxcid;
}

void NaiveBayes::print() const
{
	int outCol = OHEMaps.size()-1;
	std::map<int, std::map<int, double> >::const_iterator itr = attributesPerClass.begin();
	for (; itr != attributesPerClass.end(); ++itr)
	{
		printf("+------Class %d;%s------+\n", itr->first, OHEMaps[outCol].classAt(itr->first).c_str());
		std::map<int, double>::const_iterator itr2 = itr->second.begin();
		for (; itr2 != itr->second.end(); ++itr2)
		{
			printf("Attribute P(x=%d| C=%d) = %f\n", itr2->first, itr->first, itr2->second);
		}

		std::map<int, double>::const_iterator itr3 = classes.find(itr->first);
		if(itr3 != classes.end())
			printf("Class P(C=%d) = %f\n", itr3->first, itr3->second);
	}
}

void NaiveBayes::reset()
{
	classes.clear();
	attributesPerClass.clear();
	OHEMaps.clear();
}

std::string NaiveBayes::getClassName(int classID) const
{
    if(classID < 0)
	return "";

    int outCol = OHEMaps.size()-1;
    return OHEMaps[outCol].classAt(classID);
}
