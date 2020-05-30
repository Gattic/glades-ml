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
#ifndef _GNET
#define _GNET

#include "../Database/GList.h"
#include "socket.h"
#include <iostream>
#include <map>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/signal.h>
#include <unistd.h>
#include <vector>

class GList;

/*
 * THE ENGINE
 *
 * Handshake Client
 * HANDSHAKE_CLIENT|name\|
 *
 * Handshake Server
 * HANDSHAKE_SERVER|name|encKey\|
 *
 * Logout Server
 * LOGOUT_SERVER\|
 *
 * Logout Client
 * LOGOUT_CLIENT\|
 *
 * Bad Request
 * BAD_REQUEST\|
 *
 */

namespace GNet {

class Instance;
class Service;

// Service Arguments Class
class newServiceArgs
{
public:
	class GServer* serverInstance;
	class Instance* cInstance;
	shmea::GList sockData;
	pthread_t* sThread;
	std::string command;
};

class GServer
{
	friend Service;

	GNet::Sockets socks;

	// ip, instance pointer;instances of client connections
	std::map<std::string, Instance*>* clientInstanceList;
	std::map<std::string, Instance*>* serverInstanceList;

	int sockfd;
	Instance* localInstance;
	pthread_t* commandThread;
	pthread_t* writerThread;
	pthread_mutex_t* clientMutex;
	pthread_mutex_t* serverMutex;
	bool LOCAL_ONLY;
	bool running;
	std::map<std::string, Service*>* service_depot;

	static void* commandLauncher(void*);
	void commandCatcher(void*);
	static void* LaunchInstanceLauncher(void*);
	void LaunchInstanceHelper(void*);
	void* ListWriter(void*);
	void LaunchInstance(const std::string&, const std::string&);
	void LaunchLocalInstance(const std::string&);
	void LogoutInstance(Instance*);

public:
	static const int CONTENT_TYPE = 0;
	static const int RESPONSE_TYPE = 1;
	static const int ACK_TYPE = 2;

	GServer();
	~GServer();

	void NewService(const shmea::GList&, GNet::Instance* = NULL, int = CONTENT_TYPE, bool = false);
	Service* ServiceLookup(std::string);

	unsigned int addService(std::string, Service*);
	const bool& getRunning();
	void stop();
	void run(bool);

	bool isConnection(int, const fd_set&);
	Instance* setupNewConnection(int);
	Instance* findExistingConnectionInstance(const std::vector<Instance*>&, const fd_set&);

	int getSockFD();
	Instance* getLocalInstance();
	const std::map<std::string, Instance*>& getClientInstanceList();
	void removeClientInstance(Instance*);
	const std::map<std::string, Instance*>& getServerInstanceList();
	void removeServerInstance(Instance*);
	pthread_mutex_t* getClientMutex();
	pthread_mutex_t* getServerMutex();

	bool isNetworkingDisabled();

	static void GetDirInfo(std::string, std::vector<std::string>&,
						   std::map<std::string, std::string>&);
	static void GetURLInfo(std::string, std::string&, std::string&, std::string&, bool&);
	static std::string getWebContents(std::string);

}; // GServer

class LaunchInstanceHelperArgs
{
public:
	GServer* serverInstance;
	std::string clientName;
	std::string serverIP;
};

}; // GNet

#endif
