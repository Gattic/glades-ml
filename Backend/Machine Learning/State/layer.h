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
#ifndef _LAYER
#define _LAYER

#include "Backend/Database/GPointer.h"
#include "node.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace glades {

class Layer
{
private:
	std::vector<shmea::GPointer<Node> > children;
	std::vector<bool> dropoutFlag;
	int type;

public:
	static const int INPUT_TYPE = 0;
	static const int HIDDEN_TYPE = 1;
	static const int OUTPUT_TYPE = 2;

	// constructors and destructors
	Layer(int64_t, int, float = 0.0f);
	Layer(int);
	~Layer();

	// gets
	// Legacy/compatibility accessor:
	// Historically a single scalar bias was stored per-layer. The engine now stores biases
	// solely as per-neuron bias *edge weights* (the last edge in each gate block).
	// This method returns the average per-neuron bias for display/debug/legacy file headers.
	float getBiasWeight() const;
	int getType() const;
	unsigned int size() const;
	bool possiblePath(unsigned int) const;
	unsigned int firstValidPath() const;
	unsigned int lastValidPath() const;

	// sets
	// Legacy/compatibility setter:
	// Sets every per-neuron bias edge to the provided value.
	void setBiasWeight(float);
	void setType(int);

	// children
	const std::vector<shmea::GPointer<glades::Node> >& getChildren() const;
	Node* getNode(unsigned int);
	void setupDropout();
	void generateDropout(float);
	void clearDropout();
	void addNode(const shmea::GPointer<Node>&);
	void initWeights(int, unsigned int, int, int);
	// For gated recurrent units (GRU/LSTM): allocate per-node weights for multiple gates.
	// Layout per node: for each gate g in [0..gateCount):
	//   [prevLayerSize weights] + [1 bias]
	// Total edges per node: gateCount * (prevLayerSize + 1)
	void initGatedWeights(int prevLayerSize, unsigned int cLayerSize, int initType, int activationType, unsigned int gateCount);
	std::vector<shmea::GPointer<Node> >::iterator removeNode(Node*);
	void clean();
	void print() const;

	Node* operator[](unsigned int);

	// RNN context nodes store recurrent weights (Wh rows).
	// For gated cells, we store multiple recurrent matrices in a single context node:
	// edge layout: [gate0 hiddenSize weights][gate1 hiddenSize weights]...
	void setupContext(unsigned int gateCount = 1);
};
};

#endif
