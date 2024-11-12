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
#include "OHE.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GString.h"

using namespace glades;

glades::OHE::OHE()
{
	OHEStrings.clear();
	fMin = 0.0f;
	fMax = 0.0f;
	fMean = 0.0f;
}

glades::OHE::OHE(const OHE& ohe2)
{
	OHEStrings = ohe2.OHEStrings;
	classCount = ohe2.classCount;
	fMin = ohe2.fMin;
	fMax = ohe2.fMax;
	fMean = ohe2.fMean;
}

glades::OHE::~OHE()
{
	OHEStrings.clear();
	fMin = 0.0f;
	fMax = 0.0f;
	fMean = 0.0f;
}

void glades::OHE::addString(const char* newCharString)
{
	std::string newString(newCharString);
	addString(newString);
}

void glades::OHE::addString(const std::string& newString)
{
	// GType nodeContents(newString);
	if (!contains(newString))
	{
		OHEStrings.push_back(newString);
		classCount[newString]=1;
	}
	else
		++classCount[newString];
}

unsigned int glades::OHE::size() const
{
	return OHEStrings.size();
}

std::vector<std::string> glades::OHE::getStrings() const
{
	return OHEStrings;
}

bool glades::OHE::contains(const std::string& newString) const
{
	// check if the string is already in the vector
	std::vector<std::string>::const_iterator itr = OHEStrings.begin();
	for (; itr != OHEStrings.end(); ++itr)
	{
		if ((*itr) == newString)
			return true;
	}

	return false;
}

void glades::OHE::print() const
{
	printf("[OHE] Output:\n");
	printf("[");
	for (unsigned int i = 0; i < size(); ++i)
	{
		printf("[");
		for (unsigned int j = 0; j < size(); ++j)
		{
			// print the value
			if (i == j)
				printf("%s", OHEStrings[i].c_str());
			else
				printf("0");

			// next in a list (but not last)
			if (j < size() - 1)
				printf(",");
		}

		printf("]");
		if (i < size() - 1)
			printf("\n");
	}
	printf("]\n\n");
}

int glades::OHE::indexAt(const char* needle) const
{
	std::string needleString(needle);

	int counter = 0;
	std::vector<std::string>::const_iterator itr = OHEStrings.begin();
	for (; itr != OHEStrings.end(); ++itr)
	{
		if ((*itr) == needleString)
			return counter;
		++counter;
	}

	return -1;
}

int glades::OHE::indexAt(const std::string& needle) const
{
	std::vector<int> retVal(size(), 0);

	int counter = 0;
	std::vector<std::string>::const_iterator itr = OHEStrings.begin();
	for (; itr != OHEStrings.end(); ++itr)
	{
		if ((*itr) == needle)
			return counter;
		++counter;
	}

	return -1;
}

std::string glades::OHE::classAt(unsigned int cid) const
{
	if(cid == (unsigned int)-1)
		return "";

	if(cid >= OHEStrings.size())
		return "";

	return OHEStrings[cid];
}

std::vector<float> glades::OHE::operator[](const char* needle) const
{
	std::string needleString(needle);
	std::vector<float> retVal(size(), 0.01);

	int counter = 0;
	std::vector<std::string>::const_iterator itr = OHEStrings.begin();
	for (; itr != OHEStrings.end(); ++itr)
	{
		if ((*itr) == needleString)
		{
			retVal[counter] = 0.99;
			break;
		}

		++counter;
	}

	return retVal;
}

std::vector<float> glades::OHE::operator[](const std::string& needle) const
{
	std::vector<float> retVal(size(), 0.01);

	int counter = 0;
	std::vector<std::string>::const_iterator itr = OHEStrings.begin();
	for (; itr != OHEStrings.end(); ++itr)
	{
		if ((*itr) == needle)
		{
			retVal[counter] = 0.99;
			break;
		}

		++counter;
	}

	return retVal;
}

// ONLY SUPPORTS FIRST HOT FOUND
// EXPAND TO SUPPORT MULTIDIMENSIONALITY
std::string glades::OHE::operator[](const std::vector<int>& needle) const
{
	// check if the string is already in the vector
	int counter = 0;
	std::vector<int>::const_iterator itr = needle.begin();
	for (; itr != needle.end(); ++itr)
	{
		if ((*itr) == 1)
			return OHEStrings[counter];
		++counter;
	}

	return "";
}

std::string glades::OHE::operator[](const std::vector<float>& needle) const
{
	// check if the string is already in the vector
	float max = 0.0f;
	int counter = 0, index = -1;
	std::vector<float>::const_iterator itr = needle.begin();
	for (; itr != needle.end(); ++itr)
	{
		if ((*itr) >= max)
		{
			max = (*itr);
			index = counter;
		}
		++counter;
	}

	if (index >= 0)
		return OHEStrings[index];

	return "";
}

void glades::OHE::mapFeatureSpace(const shmea::GTable& gTable, int featureCol)
{
	for (unsigned int r = 0; r < gTable.numberOfRows(); ++r)
	{
		float cell = 0.0f;
		const shmea::GType& cCell = gTable.getCell(r, featureCol);
		if (cCell.getType() == shmea::GType::STRING_TYPE)
		{
			shmea::GString strCell = cCell;
			addString(cCell);
			continue;
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

		if (r == 0)
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
	}

	fMean /= gTable.numberOfRows();
	// Normalize
	/*std::map<std::string, double>::iterator itr = classCount.begin();
	for (; itr != classCount.end(); ++itr)
		itr->second /= gTable.numberOfRows();*/
}

float glades::OHE::standardize(float val) const
{
	// find the range of this feature
	float xRange = fMax - fMin;
	if (xRange == 0.0f)
		return 0.0f;

	return ((((val - fMin) / (xRange)) * 0.98f) + 0.01f);
}

void glades::OHE::printFeatures() const
{
	printf("[OHE] ");

	for (unsigned int i = 0; i < size(); ++i)
	{
		std::string word = OHEStrings[i];
		printf("%s ", word.c_str());
	}

	printf("\n");
}
