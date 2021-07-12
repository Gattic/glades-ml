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
#ifndef _GSAVEFOLDER
#define _GSAVEFOLDER

#include "GString.h"
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace shmea {

class GTable;
class SaveTable;

class SaveFolder
{
private:
	std::vector<SaveTable*> saveItems;
	GString dname;

	GString getPath() const;
	void addItem(SaveTable*);
	void clean();

public:
	// constructors & destructor
	SaveFolder(const GString&);
	virtual ~SaveFolder();

	SaveTable* loadItem(const GString&);
	bool deleteItem(const GString&);
	SaveTable* newItem(const GString&, const GTable&);
	void load();
	static std::vector<SaveFolder*> loadFolders();

	// gets
	GString getName() const;
	const std::vector<SaveTable*>& getItems() const;
	int size() const;
};
};

#endif
