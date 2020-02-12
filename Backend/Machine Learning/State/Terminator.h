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
#ifndef _GQL_Terminator
#define _GQL_Terminator

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace glades {

class Terminator
{
private:
	int64_t timestamp;
	int64_t epoch;
	float accuracy;

public:
	Terminator();
	~Terminator();

	// gets
	int64_t getTimestamp() const;
	int64_t getEpoch() const;
	float getAccuracy() const;

	// sets
	void setTimestamp(int64_t);
	void setEpoch(int64_t);
	void setAccuracy(float);

	// Condition check
	bool triggered(int64_t, int64_t, float) const;
};
};

#endif
