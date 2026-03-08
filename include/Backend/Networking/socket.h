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
#ifndef _GSOCKET
#define _GSOCKET

#include "../Database/GString.h"
#include "../Database/GLogger.h"
#include "../Database/GPointer.h"
#include "../Database/ServiceData.h"
#include "../Core/platform.h"
#include "../Core/GMutex.h"
#include <iostream>
#include <map>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

namespace GNet {
class GServer;
class Connection;

class Sockets
{
private:
	static const int64_t DEFAULT_KEY = 420l;
	static const shmea::GString ANYADDR;

	shmea::GString PORT;
	shmea::GMutex inMutex;
	shmea::GMutex outMutex;
	// Queue keys must be scoped to a connection to avoid cross-connection collisions.
	struct QueueKey
	{
		Connection* conn;
		int64_t serviceNum;
		QueueKey() : conn(NULL), serviceNum(0) {}
		QueueKey(Connection* c, int64_t s) : conn(c), serviceNum(s) {}
	};
	struct QueueKeyLess
	{
		bool operator()(const QueueKey& a, const QueueKey& b) const
		{
			if (a.conn < b.conn)
				return true;
			if (a.conn > b.conn)
				return false;
			return a.serviceNum < b.serviceNum;
		}
	};

	std::map<QueueKey, shmea::GPointer<shmea::ServiceData>, QueueKeyLess> inboundLists;
	std::map<QueueKey, shmea::GPointer<shmea::ServiceData>, QueueKeyLess> outboundLists;
	socket_t udpfd;
	unsigned int inboundQueueMax;
	unsigned int outboundQueueMax;

	void initSockets();
	void maybeLogMetrics(const char* where);

	// ServiceData* emptyResponseList();

public:
	static const shmea::GString LOCALHOST;

	shmea::GPointer<shmea::GLogger> logger;

	Sockets();
	Sockets(const GServer*);
	~Sockets();

	// functions
	void initSockets(const shmea::GString&);
	void closeSockets();
	const shmea::GString getPort();
	void setPort(shmea::GString);
	socket_t openServerConnection();
	socket_t openClientConnection(const shmea::GString&, const shmea::GString&);
	socket_t openUDPServerSocket();
	socket_t getUDPSocketFD() const { return udpfd; }
	void readConnection(Connection*, const socket_t&, std::vector<shmea::GPointer<shmea::ServiceData> >&);
	// Returns:
	//  1  => made progress (read bytes and/or parsed at least one complete frame)
	//  0  => no new bytes available right now (EAGAIN/EWOULDBLOCK) and no frames parsed
	// -1  => peer closed (EOF) with no remaining complete frames parsed
	// -2  => fatal I/O or protocol error (caller should logout / close connection)
	int readConnectionHelper(Connection*, const socket_t&, std::vector<shmea::GPointer<shmea::ServiceData> >&);
	int writeConnection(const Connection*, const socket_t&, shmea::ServiceData*);
	void closeConnection(const socket_t&);

	bool anyInboundLists();
	bool anyOutboundLists();

	bool readLists(Connection*);
	bool readUDPDatagram(GServer*);
	void processLists(GServer*);
	void writeLists(GServer*);
	void addResponseList(GServer*, Connection*, shmea::GPointer<shmea::ServiceData>);
	// Drop any queued inbound/outbound messages for a connection.
	// Used during logout to prevent leaks and to keep pending-send bookkeeping correct.
	void purgeConnection(Connection*);
};
};

#endif
