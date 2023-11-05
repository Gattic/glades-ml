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
#ifndef _GQL_LAYERBUILDER
#define _GQL_LAYERBUILDER

#include "Backend/Database/GTable.h"
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class Node;
class Layer;
class NNInfo;
class NetworkState;
class OHE;
class DataInput;

class LayerBuilder
{
private:
	int netType;
	std::vector<Layer*> inputLayers;
	std::vector<Layer*> layers;
	float xMin;
	float xMax;
	float xRange;
	std::vector<std::vector<std::vector<float> > > timeState;

	void seperateTables(const shmea::GTable&);
	void buildInputLayers(const NNInfo*, const DataInput*);
	void buildHiddenLayers(const NNInfo*);
	void buildOutputLayer(const NNInfo*);
	void standardizeWeights(const NNInfo*);
	float unstandardize(float);

public:
	LayerBuilder();
	LayerBuilder(int);
	~LayerBuilder();

	bool build(const NNInfo*, const DataInput*, bool = false);
	NetworkState* getNetworkStateFromLoc(unsigned int, unsigned int, unsigned int, unsigned int,
										 unsigned int);
	void setTimeState(unsigned int, unsigned int, unsigned int, float);
	unsigned int getInputLayersSize() const;
	unsigned int getLayersSize() const;
	unsigned int getLayerSize(unsigned int) const;
	unsigned int sizeOfLayer(unsigned int) const;
	float getTimeState(unsigned int, unsigned int, unsigned int) const;
	void scrambleDropout(unsigned int, float, const std::vector<float>&);
	void clearDropout();
	void print(const NNInfo*, bool = false) const;
	void clean();

	// Database
	bool load(const std::string&);
	bool save(const std::string&) const;
};
};

#endif
