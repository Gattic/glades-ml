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
#ifndef _GJSON
#define _GJSON

#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class GJson
{
private:
	static const int MAX_READ_BLOCK = 10;
	static const std::string KEY_CHARS;
	int depth;

public:
	static const int TOGGLE_KEY = 0;
	static const int TOGGLE_VALUE = 1;

	static const int VECTOR_TYPE = 0;
	static const int MAP_TYPE = 1;
	static const int PAIR_TYPE = 2;

	GJson(int, int);
	~GJson();

	int type;
	std::map<std::string, GJson*> mapValues;
	std::vector<GJson*> vecValues;
	std::string key;
	std::string value;

	static void run(std::string);
	static void print(GJson*);

	static GJson* fileToJsonHelper(FILE*, unsigned int, std::string&, int = -1, GJson* = NULL,
								   GJson* = NULL, int = -1, int = -1);
	static GJson* stringToJsonHelper(std::string&, int = -1, GJson* = NULL, GJson* = NULL, int = -1,
									 int = -1);

	static GJson* stringToJson(std::string);
	static GJson* fileToJson(std::string);

	void printHelper();
	GJson* BFS(GJson*, std::string searchKey);

	GJson* operator[](std::string);
	GJson* operator[](unsigned int);

	int getObjectSize() const;
	int getCurrentDepth() const;
	int getMaxDepth() const;
	int getMaxDepthHelper(GJson*) const;
};
};

#endif
