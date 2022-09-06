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
#ifndef _GNNETSTATE
#define _GNNETSTATE

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "Backend/Database/GPointer.h"

namespace glades {

class Layer;
class Node;

class NetworkState
{
public:
	NetworkState(unsigned int, unsigned int, unsigned int, unsigned int,
		shmea::GPointer<Layer>, shmea::GPointer<Layer>, shmea::GPointer<Node>,
		shmea::GPointer<Node>, bool, bool, bool, bool, bool, bool);

	unsigned int cInputLayerCounter;
	unsigned int cOutputLayerCounter;
	unsigned int cInputNodeCounter;
	unsigned int cOutputNodeCounter;
	shmea::GPointer<Layer> cInputLayer;
	shmea::GPointer<Layer> cOutputLayer;
	shmea::GPointer<Node> cInputNode;
	shmea::GPointer<Node> cOutputNode;
	bool firstValidInputNode, lastValidInputNode, firstValidOutputNode, lastValidOutputNode,
		validInputNode, validOutputNode;
};
};

#endif
