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
#ifndef _GCONNECTION
#define _GCONNECTION

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

namespace GNet {

class newServiceArgs;
class Service;

class Connection
{
private:
	std::string name;
	// std::string sid;//session id
	std::string ip;
	int connectionType;
	int64_t key;
	bool finished;

public:
	// member limits
	static const int KEY_LENGTH = 6;
	static const int SID_LENGTH = 12;

	// connectionType
	static const int EMPTY_TYPE = -1;
	static const int SERVER_TYPE = 0;
	static const int CLIENT_TYPE = 1;

	int sockfd;
	int64_t* overflow;
	unsigned int overflowLen;
	std::vector<Service*> sThreads; // all the active service threads

	Connection(int, int, std::string);
	Connection(const Connection&);
	~Connection();
	void finish();

	// gets
	std::string getName() const
	{
		return name;
	}
	// std::string getSID() const { return sid; }
	std::string getIP() const
	{
		return ip;
	}
	int getConnectionType() const
	{
		return connectionType;
	}
	int64_t getKey() const
	{
		return key;
	}
	bool isFinished() const
	{
		return finished;
	}

	// sets
	void setName(std::string newName)
	{
		name = newName;
	}
	// void setSID(std::string newSID) { sid=newSID; }
	void setIP(std::string newIP)
	{
		ip = newIP;
	}
	void setKey(int64_t newKey)
	{
		key = newKey;
	}

	static bool validName(const std::string&);
	static int64_t generateKey();
	// static bool validSID(const std::string&);
	// static std::string generateSID();
};
};

#endif
