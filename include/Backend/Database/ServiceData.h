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
#include "../Database/GList.h"
#include "../Database/GTable.h"
#include "../Database/GObject.h"
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
	class Serializable;

class ServiceData
{
private:

	GNet::Connection* cConnection;
	int64_t timesent;
	shmea::GString sid;
	shmea::GString command;
	shmea::GString serviceKey;
	int64_t serviceNum;
	int64_t responseServiceNum;
	int type;
	shmea::GList argList;

	shmea::GList repList;
	shmea::GTable repTable;
	shmea::GObject repObj;

public:

	static const int SID_LENGTH = 12;

	static const int TYPE_ACK = 0;//default
	static const int TYPE_LIST = 1;
	static const int TYPE_TABLE = 2;
	static const int TYPE_NETWORK_POINTER = 3;

	ServiceData(GNet::Connection*);
	ServiceData(GNet::Connection*, shmea::GString);
	ServiceData(const ServiceData&);
	virtual ~ServiceData();

	void set(shmea::GString);
	void set(shmea::GString, const shmea::GList&);
	void set(shmea::GString, const shmea::GTable&);
	void set(shmea::GString, const shmea::GObject&);
	void set(shmea::GString, const shmea::Serializable&);
	void set(const shmea::GList&);
	void set(const shmea::GTable&);
	void set(const shmea::GObject&);
	void set(const shmea::Serializable&);

	GNet::Connection* getConnection() const;
	int64_t getTimesent() const;
	shmea::GString getSID() const;
	shmea::GString getCommand() const;
	shmea::GString getServiceKey() const;
	int64_t getServiceNum() const;
	int64_t getResponseServiceNum() const;
	int getType() const;
	const GList& getArgList() const;

	void setTimesent(int64_t);
	void setSID(shmea::GString);
	void setCommand(shmea::GString);
	void setServiceKey(shmea::GString);
	void assignServiceNum();
	void assignResponseServiceNum();
	void setServiceNum(int64_t);
	void setResponseServiceNum(int64_t);
	void setType(int);
	void setArgList(const GList&);

	const GList& getList() const;
	const GTable& getTable() const;
	const GObject& getObj() const;

	void setList(const GList&);
	void setTable(const GTable&);
	void setObj(const GObject&);

	static bool validSID(const shmea::GString&);
	static shmea::GString generateSID();

	bool operator<(const ServiceData&) const;
	bool operator>(const ServiceData&) const;
};
};

#endif
