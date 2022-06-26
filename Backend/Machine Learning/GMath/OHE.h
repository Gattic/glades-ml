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
#ifndef _ONEHOTENCODING
#define _ONEHOTENCODING

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
// class GStr;
class GTable;
};

namespace glades {

class OHE
{
private:

	std::vector<std::string> OHEStrings;
	float fMin;
	float fMax;
	float fMean;

public:

	std::map<std::string, double> classCount;

	// constructors and destructors
	OHE();
	OHE(const OHE&);
	virtual ~OHE();

	// sets
	void addString(const char*);
	void addString(const std::string&);
	void mapFeatureSpace(const shmea::GTable&, int);

	// gets
	unsigned int size() const;
	std::vector<std::string> getStrings() const;
	bool contains(const std::string&) const;
	void print() const;
	void printFeatures() const;
	int indexAt(const char*) const;
	int indexAt(const std::string&) const;
	std::string classAt(unsigned int) const;
	float standardize(float) const;

	// operators
	std::vector<float> operator[](const char*) const;
	std::vector<float> operator[](const std::string&) const;
	std::string operator[](const std::vector<int>&) const;
	std::string operator[](const std::vector<float>&) const;
};
};

#endif
