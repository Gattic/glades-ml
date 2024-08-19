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
#ifndef _GNAIVEBAYES
#define _GNAIVEBAYES

#include "Backend/Database/GTable.h"
#include "../GMath/OHE.h"
#include <stdio.h>
#include <vector>
#include <map>

namespace glades {

class NaiveBayes
{
private:

	// <class id, class probility> <C, P(C)>
	std::map<int, double> classes;

	// <class id, <attribute id, probability> > <C, <x, P(x|C)> >
	std::map<int, std::map<int, double> > attributesPerClass;

	std::vector<OHE> OHEMaps;

public:

	NaiveBayes()
	{
		//
	}

	shmea::GTable import(const shmea::GTable&);
	shmea::GTable import2(const shmea::GTable&);
	void train(const shmea::GTable&);
	int predict(const shmea::GList&);
	void print() const;
	void reset();

	std::string getClassName(int) const;
};
};

#endif
