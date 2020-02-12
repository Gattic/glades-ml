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
#ifndef _GTYPES
#define _GTYPES

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace shmea {

class GType
{
private:
	char* block;
	unsigned int blockSize;
	int type;

public:
	// valueType
	static const int NULL_TYPE = -1;
	static const int BOOLEAN_TYPE = 0;
	static const int CHAR_TYPE = 1;
	static const int SHORT_TYPE = 2;
	static const int INT_TYPE = 3;
	static const int LONG_TYPE = 4;
	static const int FLOAT_TYPE = 5;
	static const int DOUBLE_TYPE = 6;
	static const int STRING_TYPE = 7;
	static const int FUNCTION_TYPE = 8;

	GType();
	GType(const GType&);
	GType(const bool&);
	GType(const char&);
	GType(const short&);
	GType(const int&);
	GType(const int64_t&);
	GType(const float&);
	GType(const double&);
	GType(const std::string&);
	GType(int, const void*, int64_t);
	~GType();

	// gets
	char* getBlockCopy() const;
	int getType() const;
	std::string getString() const;
	const char* getCString() const;
	char getChar() const;
	short getShort() const;
	int getInt() const;
	int64_t getLong() const;
	float getFloat() const;
	double getDouble() const;
	bool getBoolean() const;
	operator char() const;
	operator short() const;
	operator int() const;
	operator int64_t() const;
	operator float() const;
	operator double() const;
	operator bool() const;
	operator std::string() const;
	operator const char*() const;
	unsigned int size() const;

	// sets
	void set(int, const void*, int64_t);
	void clean();

	//= operators
	void operator=(const GType&);
	void operator=(const char&);
	void operator=(const short&);
	void operator=(const int&);
	void operator=(const int64_t&);
	void operator=(const float&);
	void operator=(const double&);
	void operator=(const char*);
	void operator=(const std::string&);
	void operator=(const bool&);

	//== operators
	bool operator==(const GType&);
	bool operator==(const char&);
	bool operator==(const short&);
	bool operator==(const int&);
	bool operator==(const int64_t&);
	bool operator==(const float&);
	bool operator==(const double&);
	bool operator==(const char*);
	bool operator==(const std::string&);
	bool operator==(const bool&);

	//!= operators
	bool operator!=(const GType&);
	bool operator!=(const char&);
	bool operator!=(const short&);
	bool operator!=(const int&);
	bool operator!=(const int64_t&);
	bool operator!=(const float&);
	bool operator!=(const double&);
	bool operator!=(const char*);
	bool operator!=(const std::string&);
	bool operator!=(const bool&);

	// helpers
	static bool isWhitespace(const char);
	static bool isWhitespace(const char*);
	static bool isWhitespace(const std::string);
	static bool isInteger(const char*);
	static bool isInteger(const std::string);
	static bool isFloat(const char*);
	static bool isFloat(const std::string);
	static bool isUpper(const char);
	static bool isUpper(const std::string);
	static char toUpper(char);
	static std::string toUpper(const std::string);
	static bool isLower(const char);
	static bool isLower(const std::string);
	static char toLower(char);
	static std::string toLower(const std::string);
	static char toggleCase(char);
	static std::string toggleCase(const std::string);
	static std::string trim(std::string);
	static std::string trim(char*);
	static std::string charTOstring(const char);
	static std::string shortTOstring(const short);
	static std::string intTOstring(const int);
	static std::string longTOstring(const int64_t);
	static std::string floatTOstring(const float);
	static std::string doubleTOstring(const double);
	static std::string datetimeTOstring(const int64_t);
	static std::string dateTOstring(const int64_t);
	static std::string timeTOstring(const int64_t);
	static unsigned int cfind(const char, const char*, const unsigned int);
};
};

#endif
