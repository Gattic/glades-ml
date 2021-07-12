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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace shmea {

class GType
{
protected:
	char* block;
	unsigned int blockSize;
	int type;

public:

	const static unsigned int npos = -1;

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
	GType(const char*);
	GType(const char*, unsigned int);
	GType(int, const void*, int64_t);
	virtual ~GType();

	// gets
	int getType() const;
	const char* c_str() const;
	char getChar() const;
	short getShort() const;
	int getInt() const;
	int64_t getLong() const;
	float getFloat() const;
	double getDouble() const;
	bool getBoolean() const;

	operator const char*() const;
	operator char() const;
	operator short() const;
	operator int() const;
	operator int64_t() const;
	operator float() const;
	operator double() const;
	operator bool() const;
	unsigned int size() const;

	// sets
	void set(int, const void*, int64_t);
	void clean();

	//= operators
	GType& operator=(const GType&);
	GType& operator=(const char&);
	GType& operator=(const short&);
	GType& operator=(const int&);
	GType& operator=(const int64_t&);
	GType& operator=(const float&);
	GType& operator=(const double&);
	GType& operator=(const char*);
	GType& operator=(const bool&);

	//== operators
	bool operator==(const GType&) const;
	bool operator==(const char&) const;
	bool operator==(const short&) const;
	bool operator==(const int&) const;
	bool operator==(const int64_t&) const;
	bool operator==(const float&) const;
	bool operator==(const double&) const;
	bool operator==(const char*) const;
	bool operator==(const bool&) const;

	//!= operators
	bool operator!=(const GType&) const;
	bool operator!=(const char&) const;
	bool operator!=(const short&) const;
	bool operator!=(const int&) const;
	bool operator!=(const int64_t&) const;
	bool operator!=(const float&) const;
	bool operator!=(const double&) const;
	bool operator!=(const char*) const;
	bool operator!=(const bool&) const;

	//< operators
	bool operator<(const GType&) const;
	bool operator<(const char&) const;
	bool operator<(const short&) const;
	bool operator<(const int&) const;
	bool operator<(const int64_t&) const;
	bool operator<(const float&) const;
	bool operator<(const double&) const;
	bool operator<(const char*) const;
	bool operator<(const bool&) const;

	//> operators
	bool operator>(const GType&) const;
	bool operator>(const char&) const;
	bool operator>(const short&) const;
	bool operator>(const int&) const;
	bool operator>(const int64_t&) const;
	bool operator>(const float&) const;
	bool operator>(const double&) const;
	bool operator>(const char*) const;
	bool operator>(const bool&) const;

	//<= operators
	bool operator<=(const GType&) const;
	bool operator<=(const char&) const;
	bool operator<=(const short&) const;
	bool operator<=(const int&) const;
	bool operator<=(const int64_t&) const;
	bool operator<=(const float&) const;
	bool operator<=(const double&) const;
	bool operator<=(const char*) const;
	bool operator<=(const bool&) const;

	//>= operators
	bool operator>=(const GType&) const;
	bool operator>=(const char&) const;
	bool operator>=(const short&) const;
	bool operator>=(const int&) const;
	bool operator>=(const int64_t&) const;
	bool operator>=(const float&) const;
	bool operator>=(const double&) const;
	bool operator>=(const char*) const;
	bool operator>=(const bool&) const;

	// Member helpers
	unsigned int cfind(char) const;
	unsigned int find(const char*, unsigned int) const;

	// Static helpers
	static unsigned int cfind(char, const char*, unsigned int);
	static unsigned int find(const char*, unsigned int, const char*, unsigned int);
};
};

#endif
