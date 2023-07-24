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
#ifndef _GMETANETWORK
#define _GMETANETWORK

#include "../State/Terminator.h"
#include "Backend/Database/GString.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class NNInfo;
class NNetwork;
class DataInput;

class MetaNetwork
{
private:
	const static int ONE_TO_ONE = 0;
	const static int ONE_TO_MANY = 1;

	shmea::GString name;
	int collectionType;
	std::vector<NNetwork*> subnets;

public:
	MetaNetwork(shmea::GString);
	MetaNetwork(NNInfo*, int = 1);
	MetaNetwork(shmea::GString metaNetName, shmea::GString nname, int = 1);
	virtual ~MetaNetwork();

	// manipulations/functions
	void clearSubnets();

	// sets
	void setName(shmea::GString);
	void addSubnet(NNInfo*);
	void addSubnet(NNetwork*);
	void addSubnet(const shmea::GString);

	// gets
	shmea::GString getName() const;
	int size() const;
	std::vector<NNetwork*> getSubnets() const;
	NNetwork* getSubnet(unsigned int);
	shmea::GString getSubnetName(unsigned int) const;
	NNetwork* getSubnetByName(shmea::GString) const;

	// verification functions
	void crossValidate(shmea::GString, DataInput*, Terminator*);
};
};

#endif
