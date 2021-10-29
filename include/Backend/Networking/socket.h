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

#include "../Database/GString.h"
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

namespace shmea {
class ServiceData;
};

namespace GNet {
class GServer;
class Connection;

class Sockets
{
private:
	static const int64_t DEFAULT_KEY = 420l;
	static const shmea::GString ANYADDR;

	shmea::GString PORT;
	pthread_mutex_t* inMutex;
	pthread_mutex_t* outMutex;
	std::queue<const shmea::ServiceData*> inboundLists;
	std::queue<const shmea::ServiceData*> outboundLists;

	void initSockets();

	// ServiceData* emptyResponseList();

public:
	static const shmea::GString LOCALHOST;

	Sockets();
	Sockets(const shmea::GString&);
	~Sockets();

	// functions
	void initSockets(const shmea::GString&);
	void closeSockets();
	const shmea::GString getPort();
	int openServerConnection();
	int openClientConnection(const shmea::GString&);
	shmea::GString reader(const int&);
	void readConnection(Connection*, const int&, std::vector<const shmea::ServiceData*>&);
	void readConnectionHelper(Connection*, const int&, std::vector<const shmea::ServiceData*>&);
	int writeConnection(const Connection*, const int&, const shmea::ServiceData*);
	void closeConnection(const int&);

	bool anyInboundLists();
	bool anyOutboundLists();

	bool readLists(Connection*);
	void processLists(GServer*, Connection*);
	void writeLists(GServer*);
	void addResponseList(GServer*, Connection*, const shmea::ServiceData*);
};
};

#endif
