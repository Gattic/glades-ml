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
#ifndef _GTABLE
#define _GTABLE

#include "GList.h"
#include "gtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <strings.h>
#include <sys/stat.h>
#include <vector>

namespace shmea {
class Serializable;

class GTable
{
private:
	friend Serializable;

	char delimiter;
	std::vector<std::string> header;
	std::vector<GList> cells;
	float xMin;
	float xMax;
	float xRange;

	std::vector<unsigned int> outputColumns; // sparse boolean array

	void importFromFile(const std::string&);
	void importFromUrl(const std::string&);
	void importFromString(const std::string&);

public:
	static const int TYPE_FILE = 0;
	static const int TYPE_URL = 1;
	static const int TYPE_STRING = 2;

	GTable();
	GTable(char);
	GTable(char, const std::vector<std::string>&);
	GTable(const std::string&, char, int);
	GTable(const GTable&);
	~GTable();
	void copy(const GTable&);

	// gets
	char getDelimiter() const;
	std::vector<std::string> getHeaders() const;
	std::string getHeader(unsigned int) const;
	GType getCell(unsigned int, unsigned int) const;
	GList getRow(unsigned int) const;
	unsigned int numberOfCols() const;
	unsigned int numberOfRows() const;
	void print() const;
	void printHeaders() const;
	GList getCol(unsigned int) const;
	GList getCol(const char*) const;
	GList getCol(const std::string&) const;
	float getMin() const;
	float getMax() const;
	float getRange() const;
	bool isOutput(unsigned int) const;
	int numOutputColumns() const;
	bool empty() const;

	// sets
	void setCell(unsigned int, unsigned int, const GType&);
	void addRow(const GList&);
	void removeRow(unsigned int);
	void clear();
	void addCol(const std::string&, const GList&,
				unsigned int = -1); // largest col, will be set to numberOfCols
	void removeCol(unsigned int);
	void swapCol(unsigned int, unsigned int);
	void moveCol(unsigned int, unsigned int);
	void append(const GTable&);
	void append(const GTable*);
	void setHeaders(const std::vector<std::string>&);
	void addHeader(unsigned int, const std::string&);
	void save(const std::string&) const;
	void toggleOutput(unsigned int);
	void clearOutputs();
	void setMin(float);
	void setMax(float);
	void setRange(float);

	// operators
	GList operator[](const char*) const;
	GList operator[](const std::string&) const;
	void operator=(const GTable&);

	static std::vector<GTable*> stratify(const GTable&, unsigned int = 10);
	static std::vector<GTable*> stratify(const std::vector<GTable*>, unsigned int = 10);
	void standardize();
	float unstandardize(float) const;
};
};

#endif
