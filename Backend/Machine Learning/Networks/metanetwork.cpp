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
#include "metanetwork.h"
#include "network.h"
#include "Backend/Database/GTable.h"
#include "../Structure/nninfo.h"
#include <fstream>

using namespace glades;

/*!
 * @brief MetaNetwork constructor
 * @param newName the lookup name of the metanetwork
 * @details user probably wants to manually call addSubnet
 */
glades::MetaNetwork::MetaNetwork(shmea::GString newName)
{
	name = newName;
	collectionType = ONE_TO_ONE;
}

/*!
 * @brief MetaNetwork constructor
 * @details initializes a MetaNetwork based on user-input data
 * @param input the input data for the neural net
 * @param networkInfo neural net information, including name
 * @param subnetCount the number of subnets in this neural net
 */
glades::MetaNetwork::MetaNetwork(NNInfo* networkInfo, int subnetCount)
{
	name = networkInfo->getName();

	if (subnetCount > 1)
		collectionType = ONE_TO_MANY;
	else
		collectionType = ONE_TO_ONE;

	// if making identical subnets (for k-folds CV or otherwise):
	// loop to fill subnets with the correct number of subnets
	for (int i = 0; i < subnetCount; ++i)
		addSubnet(networkInfo);
}

glades::MetaNetwork::MetaNetwork(shmea::GString metaNetName, shmea::GString nname, int k)
{
	name = metaNetName;
	if (k > 1)
		collectionType = ONE_TO_MANY;
	else
		collectionType = ONE_TO_ONE;
	for (int i = 0; i < k; ++i)
	{
		addSubnet(nname);
	}
}

/*!
 * @brief MetaNetwork destructor
 * @details destroys the MetaNetwork object
 */
glades::MetaNetwork::~MetaNetwork()
{
	name = "";
	clearSubnets();
}

/*!
 * @brief set MetaNetwork name
 * @details sets the name of a MetaNetwork
 * @param newName the new name for this MetaNetwork
 */
void glades::MetaNetwork::setName(shmea::GString newName)
{
	name = newName;
}

/*!
 * @brief add new subnet to the MetaNetwork
 * @details adds a new sub-network to a MetaNetwork's subnets list
 * @param networkInfo the desired network information for the subnet
 * @param inputSize the number of inputs to the subnet
 */
void glades::MetaNetwork::addSubnet(NNInfo* networkInfo)
{
	if (!networkInfo)
		return;

	shmea::GPointer<glades::NNetwork> cNetwork(new glades::NNetwork(networkInfo));
	ownedSubnets.push_back(cNetwork);
	subnets.push_back(cNetwork.get());
}

void glades::MetaNetwork::addSubnet(const shmea::GString nNetName)
{
	if (nNetName.length() <= 0)
		return;

	// Modern-only: load architecture from the model package.
	// This replaces the old NNetwork::load() shim and does not touch weights.
	const std::string modelName(nNetName.c_str());
	const std::string dir = std::string("database/models/") + modelName + "/";
	const std::string nninfoPath = dir + "nninfo.csv";
	const std::string manifestPath = dir + "manifest.txt";

	// Best-effort manifest parse (optional): netType + rngSeed.
	int netType = glades::NNetwork::TYPE_DFF;
	uint64_t seed = 0u;
	bool hasSeed = false;
	{
		std::ifstream in(manifestPath.c_str());
		if (in)
		{
			std::string line;
			// first line: magic
			std::getline(in, line);
			while (std::getline(in, line))
			{
				const size_t eq = line.find('=');
				if (eq == std::string::npos)
					continue;
				const std::string key = line.substr(0, eq);
				const std::string val = line.substr(eq + 1);
				if (key == "netType")
					netType = atoi(val.c_str());
				else if (key == "rngSeed")
				{
					seed = static_cast<uint64_t>(strtoull(val.c_str(), NULL, 10));
					hasSeed = true;
				}
			}
		}
	}

	// Load architecture (required).
	const shmea::GTable t(shmea::GString(nninfoPath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (t.numberOfRows() == 0 || t.numberOfCols() == 0)
		return;
	const glades::NNInfo info(shmea::GString(modelName.c_str()), t);

	shmea::GPointer<glades::NNetwork> cNetwork(new glades::NNetwork(&info, netType));
	if (hasSeed)
		cNetwork->setSeed(seed);

	ownedSubnets.push_back(cNetwork);
	subnets.push_back(cNetwork.get());
}

/*!
 * @brief add new subnet to the MetaNetwork
 * @details adds a new sub-network to a MetaNetwork's subnets list
 * @param networkInfo the desired network information for the subnet
 * @param inputSize the number of inputs to the subnet
 */
void glades::MetaNetwork::addSubnet(glades::NNetwork* cNetwork)
{
	if (!cNetwork)
		return;

	subnets.push_back(cNetwork);
}

/*!
 * @brief clear MetaNetwork's subnets
 * @details removes all the subnets from a MetaNetwork
 */
void glades::MetaNetwork::clearSubnets()
{
	ownedSubnets.clear();
	subnets.clear();
}

/*!
 * @brief get MetaNetwork's name
 * @details retrieves a MetaNetwork's name
 * @return the MetaNetwork's name
 */
shmea::GString glades::MetaNetwork::getName() const
{
	return name;
}

/*!
 * @brief get MetaNetwork size
 * @details retrieves the number of subnets in a MetaNetwork
 * @return the number of subnets in the MetaNetwork
 */
int glades::MetaNetwork::size() const
{
	return subnets.size();
}

/*!
 * @brief get MetaNetwork subnets
 * @details retrieves the subnets in a MetaNetwork
 * @return the MetaNetwork's subnets, a vector of NNetwork objects
 */
std::vector<glades::NNetwork*> glades::MetaNetwork::getSubnets() const
{
	return subnets;
}

/*!
 * @brief get subnet
 * @details get the subnet at a particular index
 * @param index the index of the desired subnet in the MetaNetwork
 * @return the subnet at the requested index, or NULL if the index is invalid
 */
glades::NNetwork* glades::MetaNetwork::getSubnet(unsigned int index)
{
	if (index >= subnets.size())
		return NULL;

	return subnets[index];
}

/*!
 * @brief get subnet name
 * @details get the name of a subnet at a particular index
 * @param index the index of the desired subnet in the MetaNetwork
 * @return the name of the subnet at the requested index, or empty string if the index is invalid
 */
shmea::GString glades::MetaNetwork::getSubnetName(unsigned int index) const
{
	if (index >= subnets.size())
		return "";

	return subnets[index]->getName();
}

/*!
 * @brief get subnet by name
 * @details retrieves the subnet of a given name in the MetaNetwork
 * @param newName the name of the subnet to retrieve
 * @return the subnet with the requested name, or NULL if a subnet of the requested name is not
 * present
 */
glades::NNetwork* glades::MetaNetwork::getSubnetByName(shmea::GString newName) const
{
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (subnets[i]->getName() == newName)
			return subnets[i];
	}

	// if it doesn't exist
	return NULL;
}

