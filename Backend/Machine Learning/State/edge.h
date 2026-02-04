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
#ifndef _GQL_EDGE
#define _GQL_EDGE

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class Edge
{
private:
	// edge
	float weight;
	// === SGD state ===
	// Historical implementation stored a vector<float> of per-sample deltas ("prevDelta")
	// to support minibatching. That is extremely allocation-heavy and scales poorly.
	//
	// We now store:
	// - velocity: last update step for momentum (persists across minibatches)
	// - deltaAccum: sum of update steps accumulated for current minibatch
	// - deltaCount: number of accumulated steps (for diagnostics only; averaging uses the caller's minibatchSize)
	float velocity;
	float deltaAccum;
	unsigned int deltaCount;
	bool activated;
	float activation;

public:
	Edge(int64_t, float);
	~Edge();

	// gets
	float getWeight() const;
	// Deprecated: prevDelta vectors were removed. These remain only for API compatibility.
	std::vector<float> getPrevDeltas() const;
	float getPrevDelta(unsigned int) const;
	int numPrevDeltas() const;
	float getVelocity() const;
	float getDeltaAccum() const;
	unsigned int getDeltaCount() const;
	bool getActivated() const;
	float getActivation() const;

	// sets
	void setWeight(float);
	// Accumulate a single per-sample update step into the minibatch accumulator.
	// Also updates the velocity (momentum state) to this step.
	void addPrevDelta(float);
	void setActivation(float);
	// Clears minibatch accumulation (but does NOT reset velocity/momentum).
	void clearPrevDeltas();
	void Deactivate();
};
};

#endif
