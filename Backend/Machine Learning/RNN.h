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
#ifndef _GML_RNN
#define _GML_RNN

#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class NNInfo;
class NetworkState;

class RNN : public NNetwork
{
private:
	virtual void beforeFwdEdge(const NetworkState*);
	virtual void beforeFwdNode(const NetworkState*);
	virtual void beforeFwdLayer(const NetworkState*);
	virtual void beforeFwd();
	virtual void beforeBackEdge(const NetworkState*);
	virtual void beforeBackNode(const NetworkState*);
	virtual void beforeBackLayer(const NetworkState*);
	virtual void beforeBack();

	virtual void afterFwdEdge(const NetworkState*);
	virtual void afterFwdNode(const NetworkState*, float = 0.0f);
	virtual void afterFwdLayer(const NetworkState*, float = 0.0f);
	virtual void afterFwd();
	virtual void afterBackEdge(const NetworkState*);
	virtual void afterBackNode(const NetworkState*);
	virtual void afterBackLayer(const NetworkState*);
	virtual void afterBack();

public:
	RNN();
	virtual ~RNN();
};
};

#endif
