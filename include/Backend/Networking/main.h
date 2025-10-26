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

#include "../Database/GString.h"
#include "../Database/GLogger.h"
#include "socket.h"
#include <errno.h>
#include <iostream>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/signal.h>
#include <unistd.h>
#include <vector>
/*#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>*/

namespace shmea {
class ServiceData;
};

namespace GNet {

class Connection;
class Service;
class Sockets;

// Listener interface for logout callbacks
class LogoutListener
{
public:
	virtual ~LogoutListener() {}
	virtual void onClientLogout(Connection*) = 0;
	virtual void onServerLogout(Connection*) = 0;
};

// Listener interface for login callbacks
class LoginListener
{
public:
	virtual ~LoginListener() {}
	virtual void onClientLogin(Connection*) = 0;  // when a client connects to this server
	virtual void onServerLogin(Connection*) = 0;  // when this node connects out to another server
};

// Service Arguments Class
class newServiceArgs
{
public:
	class GServer* serverInstance;
	class Connection* cConnection;
	const shmea::ServiceData* sockData;
	pthread_t* sThread;
	shmea::GString command;
	shmea::GString serviceKey;
	int stIndex;
};

class GServer
{
	friend Sockets;
	friend Service;

	shmea::GPointer<GNet::Sockets> socks;

	// Key is ip address
	//IP, [socfd, sockfd, sockfd, ...]
	std::map<shmea::GString, std::vector<int> > clientCLookUp;
	std::vector<Connection*> clientC;

	std::map<shmea::GString, std::vector<int> >serverCLookUp;
	std::vector<Connection*> serverC;

	int sockfd;
	bool cryptEnabled;
	pthread_t* commandThread;
	pthread_t* writerThread;
	pthread_mutex_t* clientMutex;
	pthread_mutex_t* serverMutex;
	pthread_mutex_t* writersMutex;
	pthread_cond_t* writersBlock;
	bool LOCAL_ONLY;
	bool running;
	std::map<shmea::GString, Service*> service_depot;
	std::map<shmea::GString, Service*> running_services;

	// Logout listener (supports class member functions)
	shmea::GPointer<LogoutListener> logoutListener;

	// Login listener
	shmea::GPointer<LoginListener> loginListener;

	static void* commandLauncher(void*);
	void commandCatcher(void*);

	static void* LaunchInstanceLauncher(void*);
	void LaunchInstanceHelper(void*);

	void wakeWriter();
	static void* ListWLauncher(void*);
	void ListWriter(void*);
	void LaunchLocalInstance(const shmea::GString&);
	void LogoutInstance(Connection*);

	int getSockFD();
	const std::vector<Connection*> getClientConnections();
	const std::vector<Connection*> getServerConnections();
	pthread_mutex_t* getClientMutex();
	pthread_mutex_t* getServerMutex();

	bool isConnection(int, const fd_set&);
	Connection* setupNewConnection(int);
	Connection* findExistingConnection(const std::vector<Connection*>&, const fd_set&);

public:

	GServer();
	~GServer();

	shmea::GPointer<shmea::GLogger> logger;

	void send(shmea::ServiceData*);

	unsigned int addService(Service*);
	Service* DoService(shmea::GString, shmea::GString = "");
	// Set the getConnection port to socks if not specified
	Connection* getConnection(shmea::GString, shmea::GString = "admin", shmea::GString = "-1");
	Connection* getConnectionFromName(shmea::GString);
	// UDP helpers
	void LaunchUDPInstance(const shmea::GString&, const shmea::GString&, const shmea::GString&);
	Connection* getOrCreateUDPConnection(const shmea::GString&, const shmea::GString&, const shmea::GString&);
	void LaunchInstance(const shmea::GString&, const shmea::GString&, const shmea::GString&);
	const bool& getRunning() const;
	shmea::GString getPort() const;
	void stop();
	void run(shmea::GString, bool);
	bool isNetworkingDisabled();
	bool isEncryptedByDefault() const;
	void enableEncryption();
	void disableEncryption();

	Connection* getLocalConnection();
	void removeClientConnection(Connection*);
	void removeServerConnection(Connection*);

	// Callbacks
	void setLogoutListener(shmea::GPointer<LogoutListener> listener) { logoutListener = listener; }
	void notifyClientLogout(Connection* c) { if (logoutListener) logoutListener->onClientLogout(c); }
	void notifyServerLogout(Connection* c) { if (logoutListener) logoutListener->onServerLogout(c); }

	void setLoginListener(shmea::GPointer<LoginListener> listener) { loginListener = listener; }
	void notifyClientLogin(Connection* c)  { if (loginListener) loginListener->onClientLogin(c); }
	void notifyServerLogin(Connection* c)  { if (loginListener) loginListener->onServerLogin(c); }
}; // GServer

class LaunchInstanceHelperArgs
{
public:
	GServer* serverInstance;
	shmea::GString clientName;
	shmea::GString serverIP;
	shmea::GString serverPort;
};

}; // GNet

#endif
