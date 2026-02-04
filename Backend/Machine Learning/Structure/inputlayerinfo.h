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
#ifndef _GQL_INPUTLAYERINFO
#define _GQL_INPUTLAYERINFO

#include "Backend/Database/GList.h"
#include "layerinfo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class InputLayerInfo : public LayerInfo
{
private:
	int batchSize;
	// Truncated backprop-through-time window length for recurrent nets (RNN/GRU/LSTM).
	// - 0 means "no truncation" (full sequence BPTT).
	// - >0 means "truncate to this many timesteps per window".
	//
	// IMPORTANT: This is intentionally separate from minibatch size.
	// Historically, this engine overloaded batchSize to mean TBPTT length for recurrent nets.
	int tbpttWindow;

public:
	// newTBPTTWindow defaults to 0 (full BPTT).
	InputLayerInfo(int, float, float, float, float, float, int, float, int newTBPTTWindow = 0);
	virtual ~InputLayerInfo();

	// gets
	int getBatchSize() const;
	int getTBPTTWindow() const;
	shmea::GList getGTableRow() const;

	// sets
	void setBatchSize(int);
	void setTBPTTWindow(int);

	int getLayerType() const;
};
};

#endif
