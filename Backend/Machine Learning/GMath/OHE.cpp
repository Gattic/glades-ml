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
#include "OHE.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GString.h"
#include <limits>

using namespace glades;

namespace {
static inline std::string key_from_gstring(const shmea::GString& s)
{
	// Normalize keys as std::string for unordered_map lookup.
	// OHEStrings remains the authoritative index->class ordering.
	return std::string(s.c_str());
}
} // namespace

glades::OHE::OHE()
{
	OHEStrings.clear();
	indexByString.clear();
	fMin = 0.0f;
	fMax = 0.0f;
	fMean = 0.0f;
}

glades::OHE::OHE(const OHE& ohe2)
{
	OHEStrings = ohe2.OHEStrings;
	classCount = ohe2.classCount;
	indexByString.clear();
	for (unsigned int i = 0; i < OHEStrings.size(); ++i)
		indexByString[key_from_gstring(OHEStrings[i])] = static_cast<int>(i);
	fMin = ohe2.fMin;
	fMax = ohe2.fMax;
	fMean = ohe2.fMean;
}

glades::OHE::~OHE()
{
	OHEStrings.clear();
	indexByString.clear();
	fMin = 0.0f;
	fMax = 0.0f;
	fMean = 0.0f;
}

void glades::OHE::addString(const shmea::GString& newString)
{
	const std::string key = key_from_gstring(newString);
	std::map<std::string, int>::iterator it = indexByString.find(key);
	if (it == indexByString.end())
	{
		OHEStrings.push_back(newString);
		indexByString[key] = static_cast<int>(OHEStrings.size() - 1u);
		classCount[newString] = 1;
	}
	else
	{
		const int idx = it->second;
		if (idx >= 0 && static_cast<unsigned int>(idx) < OHEStrings.size())
			++classCount[OHEStrings[static_cast<unsigned int>(idx)]];
		else
			++classCount[newString]; // fallback; should not happen
	}
}

void glades::OHE::setMin(float newMin)
{
	fMin = newMin;
}

void glades::OHE::setMax(float newMax)
{
	fMax = newMax;
}

void glades::OHE::setMean(float newMean)
{
	fMean = newMean;
}

unsigned int glades::OHE::size() const
{
	return OHEStrings.size();
}

float glades::OHE::getMin() const
{
	return fMin;
}

float glades::OHE::getMax() const
{
	return fMax;
}

float glades::OHE::getMean() const
{
	return fMean;
}

shmea::GVector<shmea::GString> glades::OHE::getStrings() const
{
	return OHEStrings;
}

bool glades::OHE::contains(const shmea::GString& newString) const
{
	return (indexByString.find(key_from_gstring(newString)) != indexByString.end());
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

int glades::OHE::indexAt(const shmea::GString& needle) const
{
	std::map<std::string, int>::const_iterator it = indexByString.find(key_from_gstring(needle));
	if (it == indexByString.end())
		return -1;
	return it->second;
}

shmea::GString glades::OHE::classAt(unsigned int cid) const
{
	if(cid == (unsigned int)-1)
		return "";

	if(cid >= OHEStrings.size())
		return "";

	return OHEStrings[cid];
}

shmea::GVector<float> glades::OHE::operator[](const char* needle) const
{
	shmea::GString needleString(needle);
	// Production semantics: strict one-hot encoding (0/1).
	// NOTE: The older engine used 0.99/0.01 "soft" one-hot; that is *not* a valid
	// probability distribution for multi-class softmax training and caused inconsistent
	// behavior across DataInput implementations.
	shmea::GVector<float> retVal(size(), 0.0f);

	const int idx = indexAt(needleString);
	if (idx >= 0 && static_cast<unsigned int>(idx) < retVal.size())
		retVal[static_cast<unsigned int>(idx)] = 1.0f;

	return retVal;
}

shmea::GVector<float> glades::OHE::operator[](const shmea::GString& needle) const
{
	// Strict one-hot encoding (0/1). Unknown category => all zeros.
	shmea::GVector<float> retVal(size(), 0.0f);

	const int idx = indexAt(needle);
	if (idx >= 0 && static_cast<unsigned int>(idx) < retVal.size())
		retVal[static_cast<unsigned int>(idx)] = 1.0f;

	return retVal;
}

// ONLY SUPPORTS FIRST HOT FOUND
// EXPAND TO SUPPORT MULTIDIMENSIONALITY
shmea::GString glades::OHE::operator[](const shmea::GVector<int>& needle) const
{
	// check if the string is already in the vector
	for (unsigned int counter = 0; counter < needle.size(); ++counter)
	{
		if (needle[counter] == 1)
			return OHEStrings[counter];
	}

	return "";
}

shmea::GString glades::OHE::operator[](const shmea::GVector<float>& needle) const
{
	// Argmax decode for a one-hot / probability-like vector.
	// If the vector is all zeros, treat it as "unknown" and return "".
	float max = -std::numeric_limits<float>::infinity();
	int index = -1;
	bool anyNonZero = false;
	for (unsigned int counter = 0; counter < needle.size(); ++counter)
	{
		const float v = needle[counter];
		if (v != 0.0f)
			anyNonZero = true;
		if (v > max)
		{
			max = v;
			index = static_cast<int>(counter);
		}
	}

	if (!anyNonZero || index < 0)
		return "";
	if (static_cast<unsigned int>(index) >= OHEStrings.size())
		return "";

	return OHEStrings[static_cast<unsigned int>(index)];

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
	/*std::map<shmea::GString, double>::iterator itr = classCount.begin();
	for (; itr != classCount.end(); ++itr)
		itr->second /= gTable.numberOfRows();*/
}

float glades::OHE::standardize(float val) const
{
	// find the range of this feature
	float xRange = fMax - fMin;
	if (xRange == 0.0f)
		return 0.0f;

	return ((((val - fMin) / (xRange)) * 0.99f) + 0.01f);
}

void glades::OHE::printFeatures() const
{
	printf("[OHE] ");

	for (unsigned int i = 0; i < size(); ++i)
	{
		shmea::GString word = OHEStrings[i];
		printf("%s ", word.c_str());
	}

	printf("\n");
}
