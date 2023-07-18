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
#ifndef _GNUMBERINPUT
#define _GNUMBERINPUT

#include "DataInput.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/image.h"
#include <stdio.h>
#include <vector>
#include <map>

namespace glades {

class NumberInput : public DataInput
{
public:

	// Path, Label
	shmea::GTable trainTable;
	shmea::GTable trainExpectedTable;
	shmea::GTable testTable;
	shmea::GTable testExpectedTable;

	shmea::GList emptyRow;
	shmea::GString name;
	bool loaded;

	NumberInput()
	{
		//
	    name = "";
	    loaded = false;
	    OHEMaps.clear();
	    featureIsCategorical.clear();
	    trainTable.clear();
	    trainExpectedTable.clear();
	    testTable.clear();
	    testExpectedTable.clear();
	}

	virtual ~NumberInput()
	{
	    name = "";
	    loaded = false;
	    OHEMaps.clear();
	    featureIsCategorical.clear();
	    trainTable.clear();
	    trainExpectedTable.clear();
	    testTable.clear();
	    testExpectedTable.clear();
	}

	virtual void import(shmea::GString);
	void standardizeInputTable(const shmea::GString&, int = 0);

	virtual shmea::GList getTrainRow(unsigned int) const;
	virtual shmea::GList getTrainExpectedRow(unsigned int) const;

	virtual shmea::GList getTestRow(unsigned int) const;
	virtual shmea::GList getTestExpectedRow(unsigned int) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	virtual int getType() const;
};
};

#endif
