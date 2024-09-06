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
	void beforeFwdEdge(const NetworkState&);
	void beforeFwdNode(const NetworkState&);
	void beforeFwdLayer(const NetworkState&);
	void beforeFwd();
	void beforeBackEdge(const NetworkState&);
	void beforeBackNode(const NetworkState&);
	void beforeBackLayer(const NetworkState&);
	void beforeBack();

	void afterFwdEdge(const NetworkState&);
	void afterFwdNode(const NetworkState&, float = 0.0f);
	void afterFwdLayer(const NetworkState&, float = 0.0f);
	void afterFwd();
	void afterBackEdge(const NetworkState&);
	void afterBackNode(const NetworkState&);
	void afterBackLayer(const NetworkState&);
	void afterBack();

public:
	RNN();
	~RNN();
};
};

#endif
