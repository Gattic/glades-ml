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
#ifndef _GSOCKET
#define _GSOCKET

#include "../Database/GList.h"
#include <arpa/inet.h>
#include <iostream>
#include <netdb.h>
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace GNet {
class Instance;

class Sockets
{
private:
	static const int64_t DEFAULT_KEY = 420l;

	static const std::string ANYADDR;
	static const std::string PORT;

	static int64_t* overflow;
	static unsigned int overflowLen;
	static pthread_mutex_t* inMutex;
	static pthread_mutex_t* outMutex;
	static pthread_cond_t* inWaitCond; // not used yet
	static pthread_cond_t* outWaitCond;
	static std::queue<shmea::GList> inboundLists;
	static std::queue<std::pair<Instance*, shmea::GList> > outboundLists;

	// static GList emptyResponseList();

public:
	static const std::string LOCALHOST;

	// functions
	static void initSockets();
	static void closeSockets();
	static const std::string getPort();
	static int openServerConnection();
	static int openClientConnection(const std::string&);
	static int64_t* reader(const int&, unsigned int&);
	static void readConnection(Instance*, const int&, const std::string&,
							   std::vector<shmea::GList>&);
	static void readConnectionHelper(Instance*, const int&, const std::string&,
									 std::vector<shmea::GList>&);
	static int writeConnection(const class Instance* cInstance, const int&, const shmea::GList&,
							   int);
	static void closeConnection(const int&);

	static pthread_mutex_t* getInMutex();
	static pthread_mutex_t* getOutMutex();
	static pthread_cond_t* getInWaitCond();
	static pthread_cond_t* getOutWaitCond();
	static bool anyInboundLists();
	static bool anyOutboundLists();

	static bool readLists(Instance*);
	static void processLists(Instance*);
	static bool writeLists();
	static void addResponseList(Instance*, const shmea::GList&);
};
};

#endif
