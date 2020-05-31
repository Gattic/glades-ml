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
#ifndef _GLIST
#define _GLIST

#include "GType.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class Serializable;

class GList
{
	friend Serializable;

private:
	//
	std::vector<GType> items;

	//
	void addPrimitive(int, const void*);
	void insertPrimitive(unsigned int, int, const void*);
	void addObject(int, const void*, int64_t);
	void insertObject(unsigned int, int, const void*, int64_t);

public:
	GList();
	GList(const GList&);
	GList(int, const GType&);
	~GList();

	// sets
	void copy(const GList&);
	void addChar(char);
	void insertChar(unsigned int, char);
	void addShort(short);
	void insertShort(unsigned int, short);
	void addInt(int);
	void insertInt(unsigned int, int);
	void addLong(int64_t);
	void insertLong(unsigned int, int64_t);
	void addFloat(float);
	void insertFloat(unsigned int, float);
	void addDouble(double);
	void insertDouble(unsigned int, double);
	void addBoolean(bool);
	void insertBoolean(unsigned int, bool);
	void addString(const std::string&);
	void insertString(unsigned int, const std::string&);
	void addString(const char*);
	void insertString(unsigned int, const char*);
	void addGType(const GType&);
	void insertGType(unsigned int, const GType&);
	void setGType(unsigned int, const GType&);
	void remove(unsigned int);
	void clear();

	// gets
	std::string getString(unsigned int) const;
	const char* getCString(unsigned int) const;
	char getChar(unsigned int) const;
	short getShort(unsigned int) const;
	int getInt(unsigned int) const;
	int64_t getLong(unsigned int) const;
	float getFloat(unsigned int) const;
	double getDouble(unsigned int) const;
	bool getBoolean(unsigned int) const;
	GType getGType(unsigned int) const;
	int getType(unsigned int) const;
	unsigned int size() const;
	bool empty() const;
	void print() const;

	// operators
	GType operator[](unsigned int);
	const GType operator[](unsigned int) const;
	void operator=(const GList&);
};
};

#endif
