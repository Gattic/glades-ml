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
#ifndef _LAYER
#define _LAYER

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "Backend/Database/GPointer.h"

namespace glades {

class Node;

class Layer
{
private:
	std::vector<shmea::GPointer<Node> > children;
	std::vector<bool> dropoutFlag;
	int64_t id;
	float biasWeight;
	int type;

public:
	static const int INPUT_TYPE = 0;
	static const int HIDDEN_TYPE = 1;
	static const int OUTPUT_TYPE = 2;
	static const int CONTEXT_TYPE = 3;

	// constructors and destructors
	Layer(int64_t, int, float = 0.0f);
	Layer(int);
	virtual ~Layer();

	// gets
	int64_t getID() const;
	float getBiasWeight() const;
	int getType() const;
	unsigned int size() const;
	bool possiblePath(unsigned int) const;
	unsigned int firstValidPath() const;
	unsigned int lastValidPath() const;

	// sets
	void setID(int64_t);
	void setBiasWeight(float);
	void setType(int);

	// children
	const std::vector<shmea::GPointer<Node> >& getChildren() const;
	shmea::GPointer<Node> getNode(unsigned int);
	void generateDropout(float);
	void clearDropout();
	void addNode(const shmea::GPointer<Node>&);
	void initWeights(int, unsigned int, int, int);
	void clean();
	void print() const;

	shmea::GPointer<Node> operator[](unsigned int);
};
};

#endif
