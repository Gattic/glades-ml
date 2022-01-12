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
#ifndef _GSERVICE
#define _GSERVICE

#include "../Database/GString.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

namespace shmea {
class ServiceData;
};

namespace GNet {

class GServer;
class Sockets;
class Connection;
class newServiceArgs;

class Service
{
	friend GServer;
	friend Sockets;

protected:
	// timestamp variable to store service start and end time
	static shmea::GString name;
	int64_t timeExecuted;
	pthread_t* cThread;
	bool running;

	static void* launchService(void* y);
	virtual shmea::ServiceData* execute(const shmea::ServiceData*) = 0;
	void StartService(newServiceArgs*);
	void ExitService(newServiceArgs*);

	static void ExecuteService(GServer*, const shmea::ServiceData*, Connection* = NULL);

public:
	Service();
	virtual ~Service();

	bool getRunning() const;

	virtual Service* MakeService(GServer*) const = 0;
	virtual shmea::GString getName() const = 0;
};
};

#endif
