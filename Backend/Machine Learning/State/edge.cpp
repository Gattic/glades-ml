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

glades::Edge::Edge(int64_t newID, float newWeight)
{
	id = newID;
	weight = newWeight;
	prevDelta.clear();
	activated = false;
	activation = 0.0f;
}

glades::Edge::~Edge()
{
	id = 0;
	weight = 0.0f;
	prevDelta.clear();
	activated = false;
	activation = 0.0f;
}

int64_t glades::Edge::getID() const
{
	return id;
}

float glades::Edge::getWeight() const
{
	return weight;
}

std::vector<float> glades::Edge::getPrevDeltas() const
{
	return prevDelta;
}

float glades::Edge::getPrevDelta(unsigned int index) const
{
	if (index >= prevDelta.size())
		return 0.0f;

	return prevDelta[index];
}

int glades::Edge::numPrevDeltas() const
{
	return prevDelta.size();
}

bool glades::Edge::getActivated() const
{
	return activated;
}

float glades::Edge::getActivation() const
{
	return activation;
}

void glades::Edge::setID(int64_t newID)
{
	id = newID;
}

void glades::Edge::setWeight(float newWeight)
{
	weight = newWeight;
}

void glades::Edge::addPrevDelta(float newPrevDelta)
{
	prevDelta.push_back(newPrevDelta);
}

void glades::Edge::setActivation(float newActivation)
{
	activation = newActivation;
	activated = true;
}

void glades::Edge::clearPrevDeltas()
{
	prevDelta.clear();
}

void glades::Edge::Deactivate()
{
	activation = 0.0f;
	activated = false;
}
