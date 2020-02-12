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
#ifndef _GQL2_SAVELIST
#define _GQL2_SAVELIST

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
class SaveItem;

class SaveList
{
private:
	std::vector<SaveItem*> saveItems;
	std::string dname;

	std::string getPath() const;
	void addItem(SaveItem*);
	void clean();

public:
	// constructors & destructor
	SaveList(const std::string&);
	~SaveList();

	SaveItem* loadItem(const std::string&);
	bool deleteItem(const std::string&);
	SaveItem* newItem(const std::string&, const GTable&);
	void load();
	static std::vector<SaveList*> loadFolders();

	// gets
	std::string getName() const;
	const std::vector<SaveItem*>& getItems() const;
	int size() const;
};
};

#endif
