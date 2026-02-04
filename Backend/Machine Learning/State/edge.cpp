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
#include "edge.h"

using namespace glades;

glades::Edge::Edge(int64_t newID, float newWeight)
{
	weight = newWeight;
	velocity = 0.0f;
	deltaAccum = 0.0f;
	deltaCount = 0u;
	activated = false;
	activation = 0.0f;
}

glades::Edge::~Edge()
{
	weight = 0.0f;
	velocity = 0.0f;
	deltaAccum = 0.0f;
	deltaCount = 0u;
	activated = false;
	activation = 0.0f;
}

float glades::Edge::getWeight() const
{
	return weight;
}

std::vector<float> glades::Edge::getPrevDeltas() const
{
	// Deprecated compatibility shim: historical engine stored a vector of deltas.
	// The production path no longer allocates per-sample delta vectors.
	return std::vector<float>();
}

float glades::Edge::getPrevDelta(unsigned int index) const
{
	// Deprecated compatibility shim.
	(void)index;
	return 0.0f;
}

int glades::Edge::numPrevDeltas() const
{
	// Deprecated compatibility shim: we no longer store a delta history vector.
	// Return 0 to indicate "no vector history".
	return 0;
}

float glades::Edge::getVelocity() const
{
	return velocity;
}

float glades::Edge::getDeltaAccum() const
{
	return deltaAccum;
}

unsigned int glades::Edge::getDeltaCount() const
{
	return deltaCount;
}

bool glades::Edge::getActivated() const
{
	return activated;
}

float glades::Edge::getActivation() const
{
	return activation;
}

void glades::Edge::setWeight(float newWeight)
{
	weight = newWeight;
}

void glades::Edge::addPrevDelta(float newPrevDelta)
{
	// Treat "prevDelta" as "update step" in legacy terminology.
	// Store last step for momentum and accumulate for minibatch averaging.
	velocity = newPrevDelta;
	deltaAccum += newPrevDelta;
	++deltaCount;
}

void glades::Edge::setActivation(float newActivation)
{
	activation = newActivation;
	activated = true;
}

void glades::Edge::clearPrevDeltas()
{
	// Clear minibatch accumulation; keep velocity (momentum state) intact.
	deltaAccum = 0.0f;
	deltaCount = 0u;
}

void glades::Edge::Deactivate()
{
	activation = 0.0f;
	activated = false;
}
