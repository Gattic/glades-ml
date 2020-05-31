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
#ifndef _LOGOUT_SERVER
#define _LOGOUT_SERVER

#include "../Backend/Database/GList.h"
#include "../Backend/Networking/main.h"
#include "../Backend/Networking/service.h"

class Logout_Server : public GNet::Service
{
private:
	GNet::GServer* serverInstance;

public:
	Logout_Server()
	{
		serverInstance = NULL;
	}

	Logout_Server(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~Logout_Server()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::GList execute(class GNet::Connection* cConnection, const shmea::GList& data)
	{
		shmea::GList retList;

		if (!serverInstance)
			return retList;

		printf("[SLOGOUT] %s\n", cConnection->getIP().c_str());

		// delete it from the data structure
		serverInstance->removeServerConnection(cConnection);

		// Clean up the Connection
		cConnection->finish();

		return retList;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new Logout_Server(newInstance);
	}

	std::string getName() const
	{
		return "Logout_Server";
	}
};

#endif
