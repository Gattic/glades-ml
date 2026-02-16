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
#include "ServiceData.h"
#include "../Database/Serializable.h"
#include "../Networking/connection.h"

using namespace shmea;

void shmea::delete_connection(GNet::Connection* c)
{
	delete c;
}

namespace {
// Thread-safe, process-wide monotonic counters for message identifiers.
// NOTE: This intentionally does NOT use rand(); it must be deterministic and race-free.
static pthread_mutex_t g_idMutex = PTHREAD_MUTEX_INITIALIZER;
static int64_t g_serviceCounter = 0;
static int64_t g_responseCounter = 0;
} // namespace

ServiceData::ServiceData(GNet::Connection* newConnection)
{
	cConnection = newConnection;
	connectionOwner = shmea::GPointer<GNet::Connection, shmea::delete_connection>();
	timesent = 0;
	sid = generateSID();
	command = "";
	serviceKey = "";
	type = TYPE_ACK;
	serviceNum = -1;
	responseServiceNum = -1;
}

ServiceData::ServiceData(GNet::Connection* newConnection, GString newCommand)
{
	cConnection = newConnection;
	connectionOwner = shmea::GPointer<GNet::Connection, shmea::delete_connection>();
	timesent = 0;
	sid = generateSID();
	command = newCommand;
	serviceKey = "";
	type = TYPE_ACK;
	serviceNum = -1;
	responseServiceNum = -1;
}

ServiceData::ServiceData(shmea::GPointer<GNet::Connection, shmea::delete_connection> newConnection, GString newCommand)
{
	connectionOwner = newConnection;
	cConnection = newConnection.get();
	timesent = 0;
	sid = generateSID();
	command = newCommand;
	serviceKey = "";
	type = TYPE_ACK;
	serviceNum = -1;
	responseServiceNum = -1;
}
ServiceData::ServiceData(const ServiceData& instance2)
{
	cConnection = instance2.cConnection;
	connectionOwner = instance2.connectionOwner;
	timesent = 0;
	sid = instance2.sid;
	command = instance2.command;
	serviceKey = instance2.serviceKey;
	repList = instance2.repList;
	repTable = instance2.repTable;
	repObj = instance2.repObj;
	binaryPayload = instance2.binaryPayload;
	type = instance2.type;
	serviceNum = instance2.serviceNum;
	responseServiceNum = instance2.responseServiceNum;
}

ServiceData::~ServiceData()
{
	cConnection = NULL;
	connectionOwner = shmea::GPointer<GNet::Connection, shmea::delete_connection>();
	timesent = 0;
	sid = "";
	command = "";
	serviceKey = "";
	type = TYPE_ACK;
	serviceNum = -1;
	responseServiceNum = -1;
}

void ServiceData::setConnectionOwner(shmea::GPointer<GNet::Connection, shmea::delete_connection> owner)
{
	connectionOwner = owner;
	if (owner)
		cConnection = owner.get();
}

void ServiceData::set(GString newServiceKey)
{
	serviceKey = newServiceKey;
	type = TYPE_ACK;
}

void ServiceData::set(GString newServiceKey, const GList& newList)
{
	serviceKey = newServiceKey;
	repList = newList;
	type = TYPE_LIST;
}

void ServiceData::set(GString newServiceKey, const GTable& newTable)
{
	serviceKey = newServiceKey;
	repTable = newTable;
	type = TYPE_TABLE;
}

void ServiceData::set(GString newServiceKey, const GObject& newObj)
{
	serviceKey = newServiceKey;
	repObj = newObj;
	type = TYPE_NETWORK_POINTER;
}

void ServiceData::set(GString newServiceKey, const Serializable& newNP)
{
	serviceKey = newServiceKey;
	repObj = newNP.serialize();
	type = TYPE_NETWORK_POINTER;
}

void ServiceData::set(const GList& newList)
{
	serviceKey = "";
	repList = newList;
	type = TYPE_LIST;
}

void ServiceData::set(const GTable& newTable)
{
	serviceKey = "";
	repTable = newTable;
	type = TYPE_TABLE;
}

void ServiceData::set(const GObject& newObj)
{
	serviceKey = "";
	repObj = newObj;
	type = TYPE_NETWORK_POINTER;
}

void ServiceData::set(const Serializable& newNP)
{
	serviceKey = "";
	repObj = newNP.serialize();
	type = TYPE_NETWORK_POINTER;
}

void ServiceData::set(GString newServiceKey, const char* data, unsigned int size)
{
	serviceKey = newServiceKey;
	binaryPayload = GString(data, size);
	type = TYPE_BINARY;
}

void ServiceData::setBinaryPayload(const char* data, unsigned int size)
{
	binaryPayload = GString(data, size);
	type = TYPE_BINARY;
}

const GString& ServiceData::getBinaryPayload() const
{
	return binaryPayload;
}

unsigned int ServiceData::getBinaryPayloadSize() const
{
	return binaryPayload.length();
}

const GList& ServiceData::getList() const
{
	return repList;
}

const GTable& ServiceData::getTable() const
{
	return repTable;
}

const GObject& ServiceData::getObj() const
{
	return repObj;
}

void ServiceData::setList(const GList& newList)
{
	repList = newList;
}

void ServiceData::setTable(const GTable& newTable)
{
	repTable = newTable;
}

void ServiceData::setObj(const GObject& newObj)
{
	repObj = newObj;
}

GNet::Connection* ServiceData::getConnection() const
{
	return cConnection;
}

int64_t ServiceData::getTimesent() const
{
	return timesent;
}

GString ServiceData::getSID() const
{
	return sid;
}

GString ServiceData::getCommand() const
{
	return command;
}

GString ServiceData::getServiceKey() const
{
	return serviceKey;
}

int64_t ServiceData::getServiceNum() const
{
	return serviceNum;
}

int64_t ServiceData::getResponseServiceNum() const
{
	return responseServiceNum;
}

int ServiceData::getType() const
{
	return type;
}

const GList& ServiceData::getArgList() const
{
	return argList;
}

void ServiceData::setTimesent(int64_t newTS)
{
	timesent = newTS;
}

void ServiceData::setSID(GString newSID)
{
	sid = newSID;
}

void ServiceData::setCommand(GString newCommand)
{
	command = newCommand;
}

void ServiceData::setServiceKey(GString newServiceKey)
{
	serviceKey = newServiceKey;
}


void ServiceData::assignServiceNum()
{
	// Generate a unique id even when multiple threads are sending concurrently.
	// Start at 1 (0 is treated as "unset" in some contexts).
	pthread_mutex_lock(&g_idMutex);
	serviceNum = ++g_serviceCounter;
	pthread_mutex_unlock(&g_idMutex);
}

void ServiceData::assignResponseServiceNum()
{
	// The previous implementation (serviceNum + 1) was not unique and was not safe
	// under concurrency. Generate an independent unique id.
	pthread_mutex_lock(&g_idMutex);
	responseServiceNum = ++g_responseCounter;
	pthread_mutex_unlock(&g_idMutex);
}

void ServiceData::setServiceNum(int64_t newServiceNum)
{
	serviceNum = newServiceNum;
}

void ServiceData::setResponseServiceNum(int64_t newResponseServiceNum)
{
	responseServiceNum = newResponseServiceNum;
}

void ServiceData::setType(int newType)
{
	type = newType;
}

void ServiceData::setArgList(const GList& newList)
{
	argList = newList;
}

bool ServiceData::validSID(const GString& testSID)
{
	if(testSID.length() != SID_LENGTH) return false;

	const GString options=
		"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+[{]};:,<.>/?";

	for(unsigned int i=0;i<testSID.length();++i)
	{
		int breakPoint=options.cfind(testSID[i]);
		if(breakPoint == -1) return false;
	}

	return true;
}

GString ServiceData::generateSID()
{
	const GString options=
		"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+[{]};:,<.>/?";
	GString newSID="";
	do
	{
		newSID="";
		for(unsigned int i=0;i<SID_LENGTH;++i)
		{
			int newIndex=rand()%options.length();
			char newChar=options[newIndex];
			newSID+=newChar;
		}
	} while(!validSID(newSID));// && check if its in the data structure to avoid hijacking

	return newSID;
}

bool ServiceData::operator<(const ServiceData& sd2) const
{
	return getServiceNum() < sd2.getServiceNum();
}

bool ServiceData::operator>(const ServiceData& sd2) const
{
	return getServiceNum() > sd2.getServiceNum();
}
