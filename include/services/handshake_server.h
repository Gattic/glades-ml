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
#ifndef _HANDSHAKE_SERVER
#define _HANDSHAKE_SERVER

#include "../Backend/Database/GString.h"
#include "../Backend/Database/ServiceData.h"
#include "../Backend/Networking/main.h"
#include "../Backend/Networking/service.h"
#include "../Backend/Networking/socket.h"

class Handshake_Server : public GNet::Service
{
private:
	GNet::GServer* serverInstance;

public:
	Handshake_Server()
	{
		serverInstance = NULL;
	}

	Handshake_Server(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~Handshake_Server()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		// Log the client into the server
		class GNet::Connection* destination = data->getConnection();

		if (!serverInstance)
			return NULL;

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		const shmea::GList* cList = data->getList();
		if ((!cList) || cList->size() < 1)
			return NULL;

		// Check the characters in the name
		shmea::GString clientName = cList->getString(0);
		if (GNet::Connection::validName(clientName))
			destination->setName(clientName);

		// generate a new key for the sockets
		// if (destination->getIP() != GNet::Sockets::LOCALHOST)
		{
			int64_t newKey = GNet::Connection::generateKey();

			// tell the client the good news
			shmea::GList* wData = new shmea::GList();
			wData->addString(destination->getName());
			wData->addLong(newKey);

			shmea::ServiceData* cData = new shmea::ServiceData(destination, "Handshake_Client", wData);
			// GNet::Service::ExecuteService(serverInstance, wData, destination);
			serverInstance->send(cData);

			// Set the new Connection key
			destination->setKey(newKey);
		}

		return NULL;
	}

	shmea::GString getName() const
	{
		return "Handshake_Server";
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new Handshake_Server(newInstance);
	}
};

#endif
