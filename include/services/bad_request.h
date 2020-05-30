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
#ifndef _BAD_REQUEST
#define _BAD_REQUEST

#include "../Backend/Database/GList.h"
#include "../Backend/Networking/main.h"
#include "../Backend/Networking/service.h"

class Bad_Request : public GNet::Service
{
private:
	GNet::GServer* serverInstance;

public:
	Bad_Request()
	{
		serverInstance = NULL;
	}

	Bad_Request(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~Bad_Request()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::GList execute(class GNet::Instance* cInstance, const shmea::GList& data)
	{
		shmea::GList retList;
		if (!serverInstance)
			return retList;

		printf("[SRVC] Bad Request\n");

		return retList;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new Bad_Request(newInstance);
	}

	std::string getName() const
	{
		return "Bad_Request";
	}
};

#endif
