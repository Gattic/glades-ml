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
#include "main.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Networking/main.h"
#include "GMath/gmath.h"
#include "Structure/nninfo.h"
#include "Networks/metanetwork.h"
#include "Networks/network.h"
#include "Networks/training_callbacks.h"
#include "DataObjects/ImageInput.h"

#include <utility>

using namespace glades;

bool glades::doesDatabaseExist()
{
	struct stat buffer;
	return stat("database", &buffer) == 0 && S_ISDIR(buffer.st_mode);
}

void glades::createDatabase()
{
#if defined(_WIN32)
	_mkdir("database");
#else
	mkdir("database", 0777);
#endif
}

void glades::init()
{
	if (!doesDatabaseExist())
		createDatabase();
}

bool glades::saveNeuralNetwork(glades::NNetwork* newNet)
{
	if (!newNet)
		return false;

	const glades::NNetworkStatus status =
		newNet->saveModel(std::string(newNet->getName().c_str()));
	if (!status.ok())
	{
		char buffer[256];
		sprintf(buffer, "Unable to save \"%s\"", newNet->getName().c_str());
		puts(buffer);
		return false;
	}
	return true;
}

namespace {

shmea::GPointer<glades::MetaNetwork> RunMetaNetwork(
	shmea::GPointer<glades::MetaNetwork> network,
	glades::DataInput* input,
	glades::ITrainingCallbacks* callbacks,
	GNet::GServer* server,
	GNet::Connection* connection,
	bool training)
{
	if (!network || !input)
		return {};

	const auto subnets = network->getSubnets();
	for (glades::NNetwork* subnet : subnets)
	{
		if (!subnet)
			return {};
		if (server && connection)
			subnet->setServer(server, connection);
		const glades::NNetworkStatus status = callbacks
			? (training ? subnet->train(input, callbacks)
			            : subnet->test(input, callbacks))
			: (training ? subnet->train(input) : subnet->test(input));
		if (!status.ok())
		{
			printf("[NN] %s failed: %s\n",
			       training ? "Train" : "Test", status.message.c_str());
			return {};
		}
	}
	return network;
}

shmea::GPointer<glades::MetaNetwork> MakeMetaNetwork(glades::NNInfo* info)
{
	if (!info)
		return {};
	auto network = shmea::make_gpointer<glades::MetaNetwork>(info->getName());
	network->addSubnet(info);
	return network;
}

shmea::GPointer<glades::MetaNetwork> MakeMetaNetwork(glades::NNetwork* subnet)
{
	if (!subnet)
		return {};
	auto network = shmea::make_gpointer<glades::MetaNetwork>(subnet->getName());
	network->addSubnet(subnet);
	return network;
}

} // namespace

shmea::GPointer<glades::MetaNetwork> glades::train(
	NNInfo* info, DataInput* input, GNet::GServer* server,
	GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(info), input, nullptr, server, connection, true);
}

shmea::GPointer<glades::MetaNetwork> glades::train(
	NNetwork* subnet, DataInput* input, GNet::GServer* server,
	GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(subnet), input, nullptr, server, connection, true);
}

shmea::GPointer<glades::MetaNetwork> glades::train(
	shmea::GPointer<MetaNetwork> network, DataInput* input,
	GNet::GServer* server, GNet::Connection* connection)
{
	return RunMetaNetwork(
		std::move(network), input, nullptr, server, connection, true);
}

shmea::GPointer<glades::MetaNetwork> glades::test(
	NNInfo* info, DataInput* input, GNet::GServer* server,
	GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(info), input, nullptr, server, connection, false);
}

shmea::GPointer<glades::MetaNetwork> glades::test(
	NNetwork* subnet, DataInput* input, GNet::GServer* server,
	GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(subnet), input, nullptr, server, connection, false);
}

shmea::GPointer<glades::MetaNetwork> glades::test(
	shmea::GPointer<MetaNetwork> network, DataInput* input,
	GNet::GServer* server, GNet::Connection* connection)
{
	return RunMetaNetwork(
		std::move(network), input, nullptr, server, connection, false);
}

shmea::GPointer<glades::MetaNetwork> glades::train(
	NNetwork* subnet, DataInput* input, ITrainingCallbacks* callbacks,
	GNet::GServer* server, GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(subnet), input, callbacks, server, connection, true);
}

shmea::GPointer<glades::MetaNetwork> glades::test(
	NNetwork* subnet, DataInput* input, ITrainingCallbacks* callbacks,
	GNet::GServer* server, GNet::Connection* connection)
{
	return RunMetaNetwork(
		MakeMetaNetwork(subnet), input, callbacks, server, connection, false);
}
