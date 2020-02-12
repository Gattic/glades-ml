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
#ifndef _GQL2_SAVEITEM
#define _GQL2_SAVEITEM

#include "gtable.h"
#include <fstream>
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace shmea {
class SaveItem
{
private:
	int64_t id;
	std::string dname;
	std::string name;
	GTable value;

	std::string getPath() const;

protected:
	friend class SaveList;

	// Load functions
	void loadByID(int64_t);

	// Save Functions
	void saveByID(const GTable&);

	void clean();

public:
	// constructors & destructor
	SaveItem(const std::string&, const std::string&);
	~SaveItem();

	// Database operations
	void loadByName();
	void saveByName(const GTable&) const;
	bool deleteByName();

	// gets
	int64_t getID() const;
	std::string getName() const;
	GTable getTable() const;
	void print() const;
};
};

#endif
