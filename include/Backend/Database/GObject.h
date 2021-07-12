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
#ifndef _GOBJECT
#define _GOBJECT

#include "GTable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class Serializable;

class GObject
{
	friend Serializable;

private:
	//
	GTable members;
	std::vector<GTable> memberTables;

public:
	GObject();
	GObject(const GObject&);
	~GObject();

	// sets
	void copy(const GObject&);
	void addTable(const GTable&);
	void insertTable(unsigned int, const GTable&);
	void setTable(unsigned int, const GTable&);
	void setMembers(const GTable&);
	void remove(unsigned int);
	void clear();

	// gets
	GTable getTable(unsigned int) const;
	const GTable& getMembers() const;
	unsigned int size() const;
	bool empty() const;

	// operators
	GTable operator[](unsigned int);
	const GTable operator[](unsigned int) const;
	void operator=(const GObject&);
};
};

#endif
