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
class GServer;
class Connection;

class Sockets
{
private:
	static const int64_t DEFAULT_KEY = 420l;
	static const std::string ANYADDR;

	std::string PORT;
	int64_t* overflow; // dont free this
	unsigned int overflowLen;
	pthread_mutex_t* inMutex;
	pthread_mutex_t* outMutex;
	std::queue<shmea::GList> inboundLists;
	std::queue<std::pair<Connection*, shmea::GList> > outboundLists;

	void initSockets();

	// GList emptyResponseList();

public:
	static const std::string LOCALHOST;

	Sockets();
	Sockets(const std::string&);
	~Sockets();

	// functions
	void initSockets(const std::string&);
	void closeSockets();
	const std::string getPort();
	int openServerConnection();
	int openClientConnection(const std::string&);
	int64_t* reader(const int&, unsigned int&);
	void readConnection(Connection*, const int&, const std::string&, std::vector<shmea::GList>&);
	void readConnectionHelper(Connection*, const int&, const std::string&,
							  std::vector<shmea::GList>&);
	int writeConnection(const Connection*, const int&, const shmea::GList&, int);
	void closeConnection(const int&);

	bool anyInboundLists();
	bool anyOutboundLists();

	bool readLists(Connection*);
	void processLists(GServer*, Connection*);
	void writeLists(GServer*);
	void addResponseList(GServer*, Connection*, const shmea::GList&);
};
};

#endif
