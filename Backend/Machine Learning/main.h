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
#ifndef _GLADESML
#define _GLADESML

#include <algorithm>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/signal.h>
#include <sys/stat.h>
#include <vector>

#include "Backend/Database/GPointer.h"
#include "Networks/metanetwork.h" // ensure MetaNetwork is complete for GPointer default deleter

namespace shmea {
class GTable;
};

namespace GNet {
class GServer;
class Connection;
};

namespace glades {
class NNInfo;
class NNetwork;
class RNN;
class DataInput;
class ITrainingCallbacks;

void init();
RNN* getRNN(const std::string&);
bool saveNeuralNetwork(NNetwork*);

// Machine-learning operations return explicit shared ownership. Input pointers
// are non-owning borrows; existing MetaNetwork operations retain ownership by
// accepting and returning GPointer.
shmea::GPointer<MetaNetwork> train(
	NNInfo*, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> train(
	NNetwork*, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> train(
	shmea::GPointer<MetaNetwork>, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> test(
	NNInfo*, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> test(
	NNetwork*, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> test(
	shmea::GPointer<MetaNetwork>, DataInput*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);

// Callback-aware API.
shmea::GPointer<MetaNetwork> train(
	NNetwork*, DataInput*, ITrainingCallbacks*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);
shmea::GPointer<MetaNetwork> test(
	NNetwork*, DataInput*, ITrainingCallbacks*, GNet::GServer* = nullptr,
	GNet::Connection* = nullptr);

// Database Setup
bool doesDatabaseExist();
void createDatabase();
};

#endif
