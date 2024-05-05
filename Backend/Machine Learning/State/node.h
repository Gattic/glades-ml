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
#ifndef _GQL_NODE
#define _GQL_NODE

#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>

class GType;

namespace glades {

class Edge;

// Makes up the graph
class Node
{
private:
	int64_t id;
	float weight;
	float errorDer;
	float activationScalar;

	// ml
	std::vector<glades::Edge*> edges;
	pthread_mutex_t* activationMutex;

public:
	static const int INIT_EMPTY = 0;
	static const int INIT_RANDOM = 1;
	static const int INIT_POSRAND = 2;
	static const int INIT_XAVIER = 3;
	static const int INIT_POSXAVIER = 4;

	Node();
	Node(GType*);
	Node(const Node&); // copy contstructor
	~Node();
	void copy(const Node&);

	// gets
	int64_t getID() const;
	float getWeight() const;
	float getDropout() const;
	float getEdgeWeight(unsigned int) const;
	int64_t getEdgeID(unsigned int) const;
	float getActivation() const;
	float getActivationScalar() const;
	float getErrDer() const;
	unsigned int numEdges() const;
	std::vector<float> getPrevDeltas(unsigned int) const;
	float getLastPrevDelta(unsigned int) const;

	// sets
	void setID(int64_t);
	void setWeight(float);
	void setEdges(const std::vector<glades::Edge*>&);
	void setEdgeWeight(unsigned int, float);
	void setActivation(unsigned int, float);
	void setActivationScalar(float);
	void clearActivation();
	void adjustErrDer(float);
	void clearErrDer();
	void addPrevDelta(unsigned int, float);
	void clearPrevDeltas(unsigned int);
	void clean();
	void print() const;

	// weights functions
	void initWeights(unsigned int, int);
	void initWeights(unsigned int, float[], unsigned int, int, int);
	void getDelta(unsigned int, float, float, float, float, float, float);
	void applyDeltas(unsigned int, int);
};
};

#endif
