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
#ifndef _GSERVICEDATA
#define _GSERVICEDATA

#include "../Database/GString.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

namespace GNet {
class Connection;
};

namespace shmea {
	class GList;
	class GTable;
	class GObject;
	class Serializable;

class ServiceData
{
private:

	GNet::Connection* cConnection;
	shmea::GString sid;
	shmea::GString command;
	int type;

	shmea::GList* repList;
	shmea::GTable* repTable;
	shmea::GObject* repObj;

public:

	static const int SID_LENGTH = 12;

	static const int TYPE_ACK = 0;//default
	static const int TYPE_LIST = 1;
	static const int TYPE_TABLE = 2;
	static const int TYPE_NETWORK_POINTER = 3;

	ServiceData(GNet::Connection*);
	ServiceData(GNet::Connection*, shmea::GString);
	ServiceData(GNet::Connection*, shmea::GString, shmea::GList*);
	ServiceData(GNet::Connection*, shmea::GString, shmea::GTable*);
	ServiceData(GNet::Connection*, shmea::GString, shmea::Serializable*);
	ServiceData(const ServiceData&);
	virtual ~ServiceData();

	GNet::Connection* getConnection() const;
	shmea::GString getSID() const;
	shmea::GString getCommand() const;
	int getType() const;

	void setSID(shmea::GString);
	void setCommand(shmea::GString);
	void setType(int);

	const GList* getList() const;
	const GTable* getTable() const;
	const GObject* getObj() const;

	void setList(GList*);
	void setTable(GTable*);
	void setObj(GObject*);

	static bool validSID(const shmea::GString&);
	static shmea::GString generateSID();
};
};

#endif
