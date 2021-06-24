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
#ifndef _HANDSHAKE_CLIENT
#define _HANDSHAKE_CLIENT

#include "../Backend/Database/GString.h"
#include "../Backend/Database/GList.h"
#include "../Backend/Database/ServiceData.h"
#include "../Backend/Networking/connection.h"
#include "../Backend/Networking/service.h"

class Handshake_Client : public GNet::Service
{
private:
	GNet::GServer* serverInstance;

public:
	Handshake_Client()
	{
		serverInstance = NULL;
	}

	Handshake_Client(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~Handshake_Client()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		// Log the server into the client
		class GNet::Connection* destination = data->getConnection();

		if (!serverInstance)
			return NULL;

		const shmea::GList* cList = data->getList();
		if (!cList)
			return NULL;

		if ((data->getType() != shmea::ServiceData::TYPE_LIST) || cList->size() < 2)
			return NULL;

		// Check the characters in the name
		shmea::GString clientName = cList->getString(0);
		if (!GNet::Connection::validName(clientName))
			clientName = "";
		destination->setName(clientName);

		// get the new encryption key
		int64_t newKey = cList->getLong(1);

		// Set the new Connection key
		destination->setKey(newKey);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new Handshake_Client(newInstance);
	}

	shmea::GString getName() const
	{
		return "Handshake_Client";
	}
};

#endif
