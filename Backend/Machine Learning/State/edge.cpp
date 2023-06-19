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
#include "edge.h"

using namespace glades;

Edge::Edge(int64_t newID, float newWeight)
{
	id = newID;
	weight = newWeight;
	prevDelta.clear();
	activated = false;
	activation = 0.0f;
}

Edge::~Edge()
{
	id = 0;
	weight = 0.0f;
	prevDelta.clear();
	activated = false;
	activation = 0.0f;
}

int64_t Edge::getID() const
{
	return id;
}

float Edge::getWeight() const
{
	return weight;
}

std::vector<float> Edge::getPrevDeltas() const
{
	return prevDelta;
}

float Edge::getPrevDelta(unsigned int index) const
{
	if (index >= prevDelta.size())
		return 0.0f;

	return prevDelta[index];
}

int Edge::numPrevDeltas() const
{
	return prevDelta.size();
}

bool Edge::getActivated() const
{
	return activated;
}

float Edge::getActivation() const
{
	return activation;
}

void Edge::setID(int64_t newID)
{
	id = newID;
}

void Edge::setWeight(float newWeight)
{
	weight = newWeight;
}

void Edge::addPrevDelta(float newPrevDelta)
{
	prevDelta.push_back(newPrevDelta);
}

void Edge::setActivation(float newActivation)
{
	activation = newActivation;
	activated = true;
}

void Edge::clearPrevDeltas()
{
	prevDelta.clear();
}

void Edge::Deactivate()
{
	activation = 0.0f;
	activated = false;
}
