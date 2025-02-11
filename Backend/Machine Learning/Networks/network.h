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
#ifndef _NNETWORK
#define _NNETWORK

#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "../State/Terminator.h"
#include "../GMath/cmatrix.h"
#include "../State/LayerBuilder.h"
#include "bayes.h"
#include <algorithm>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>

class Point2;

namespace shmea {
class GTable;
};

namespace GNet {
class GServer;
class Connection;
};

namespace glades {

class DataInput;
class NNInfo;
class Layer;
class Node;
class NetworkState;
class LayerBuilder;
class CMatrix;
class MetaNetwork;

class NNetwork
{
private:
	friend MetaNetwork;

	DataInput* di;
	NNInfo* skeleton;
	LayerBuilder meat;
	CMatrix confusionMatrix;
	GNet::GServer* serverInstance;
	GNet::Connection* cConnection;
	glades::NaiveBayes bModel;

	bool running;
	int netType;
	int epochs;
	bool saveInstance;
	float overallTotalError;
	float overallTotalAccuracy;
	float overallClassAccuracy;
	float overallClassPrecision;
	float overallClassRecall;
	float overallClassSpecificity;
	float overallClassF1;
	int minibatchSize;
	int64_t id;

	bool firstRunActivation;

	// for tables & graphs
	shmea::GList learningCurve;
	std::vector<Point2*> rocCurve;
	shmea::GList results;
	shmea::GTable nbRecord;
	//Only for sending on the network
	shmea::GList cNodeActivations;

	void run(DataInput*, int);
	void SGDHelper(unsigned int, int); // Stochastic Gradient Descent

	void ForwardPass(unsigned int, int, int, unsigned int, unsigned int);
	void BackPropagation(unsigned int, int, int, unsigned int, unsigned int);

public:
	static const int TYPE_DFF = 0;
	static const int TYPE_RNN = 1;

	static const int RUN_TRAIN = 0;
	static const int RUN_TEST = 1;
	static const int RUN_VALIDATE = 2;

	Terminator terminator;

	NNetwork(int=TYPE_DFF);
	NNetwork(NNInfo*, int=TYPE_DFF);
	virtual ~NNetwork();
	int64_t getCurrentTimeMilliseconds() const;
	bool getRunning() const;
	int getEpochs() const;
	void stop();

	// Database
	bool load(const shmea::GString&);
	bool save() const;
	void setServer(GNet::GServer*, GNet::Connection*);

	// Stochastic Gradient Descent
	void train(DataInput*);
	void test(DataInput*);

	int64_t getID() const;
	shmea::GString getName() const;
	NNInfo* getNNInfo();
	float getAccuracy() const;

	// graphing
	shmea::GList getLearningCurve() const;
	// const std::vector<Point2*>& getROCCurve() const;
	shmea::GList getResults() const;
	void clean();
	void resetGraphs();
};
};

#endif
