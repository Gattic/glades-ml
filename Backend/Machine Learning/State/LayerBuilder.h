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
#ifndef _GQL_LAYERBUILDER
#define _GQL_LAYERBUILDER

#include "Backend/Database/GPointer.h"
#include "Backend/Database/GTable.h"
#include "layer.h"
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>

namespace glades {

class Node;
class NNInfo;
class NetworkState;
class OHE;
class DataInput;

class LayerBuilder
{
private:
	int netType;
	// Build-time error reporting.
	// When build() returns false, this is set to a human-readable reason.
	std::string lastError;
	void setError(const std::string& msg) { lastError = msg; }
	// Input handling:
	// Historically, this builder allocated one full input Layer per training row.
	// That is extremely memory-inefficient and destroys cache locality.
	//
	// We now keep a single reusable input layer of size = featureCount and overwrite its
	// node weights from the requested training row on demand.
	shmea::GPointer<Layer> inputLayer;
	unsigned int inputRowCount;
	unsigned int inputFeatureCount;
	const DataInput* dataInput; // non-owning; valid during build/run
	std::vector<shmea::GPointer<Layer> > layers;
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

    bool saveLayer(Layer*, std::ofstream&) const;
    bool loadLayer(Layer*, unsigned int, unsigned int, std::ifstream&);

public:
	LayerBuilder();
	LayerBuilder(int);
	~LayerBuilder();

	// Dataset attachment (lifetime safety)
	//
	// LayerBuilder needs a DataInput only to materialize input rows on demand in getInputLayer().
	// Historically it stored a raw pointer set during build() and kept it indefinitely.
	// That is unsafe because the caller typically owns DataInput and may delete it after a run.
	//
	// The training driver (`Trainer`) is responsible for attaching the active dataset for the
	// duration of a train/test run, and detaching it afterwards.
	void attachDataInput(const DataInput* di) { dataInput = di; }
	void detachDataInput() { dataInput = NULL; }
	const DataInput* getAttachedDataInput() const { return dataInput; }

	bool build(const NNInfo*, const DataInput*, bool = false);
	// Overload used by NNetwork: selects build behavior by net type (DFF/RNN/GRU/LSTM).
	// This intentionally coexists with the historical bool overload (standardizeWeightsFlag).
	bool build(const NNInfo*, const DataInput*, int /*netType*/, bool /*standardizeWeightsFlag*/ = false);
	const std::string& getLastError() const { return lastError; }
	void rebuildInputLayers(const NNInfo*, const DataInput*);
	// Input materialization:
	// The builder holds a single reusable input layer; this call overwrites its node weights
	// from the requested dataset row on demand.
	//
	// IMPORTANT: Training and evaluation can use different dataset splits. Callers should
	// use the explicit split overload in any run-loop code.
	enum InputSplit
	{
		SPLIT_TRAIN = 0,
		SPLIT_TEST = 1
	};
	// Backward-compatible default: historically always used the training split.
	Layer* getInputLayer(unsigned int, unsigned int);
	// Explicit split selection.
	Layer* getInputLayer(unsigned int inputRowCounter, unsigned int cInputLayerCounter, InputSplit split);
	Layer* getOutputLayer(unsigned int);
	Node* getInputNode(Layer*, unsigned int);
	Node* getOutputNode(Layer*, unsigned int);

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

	// Getters
        shmea::GList getWeights() const;
	shmea::GList getActivations();
	void addBiasWeights(shmea::GList&) const;

	// RNN helpers (context nodes)
	void resetContextState(float value = 0.0f);
	void updateContextFromHiddenActivations();

	// Database
	bool load(const std::string&);
	bool save(const std::string&) const;

	// Unified model persistence helpers.
	//
	// Historical API saved/loaded weights in `database/nn-state/<fileName>`.
	// For production model packaging we also support reading/writing to an explicit path.
	bool saveStateToFile(const std::string& filePath) const;
	bool loadStateFromFile(const NNInfo* skeleton, const std::string& filePath);

	// Backwards-compatible wrappers (legacy location: `database/nn-state/<fileName>`).
	bool saveState(const char* fileName) const;
	bool loadState(const NNInfo* skeleton, const char* fileName);
};
};

#endif
