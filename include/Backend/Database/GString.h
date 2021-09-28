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
#ifndef _GSTRING
#define _GSTRING

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdexcept>
#include "GType.h"

namespace shmea {

class GString : public GType
{
private:

public:
	GString();
	GString(const GString&);
	GString(const GType&);
	GString(const char*);
	GString(const char*, unsigned int);

	// Our GType constructors
	GString(const bool&);
	GString(const char&);
	GString(const short&);
	GString(const int&);
	GString(const int64_t&);
	GString(const float&);
	GString(const double&);

	virtual ~GString();

	// gets
	unsigned int length() const;

	//operators
	const char& operator[](const unsigned int&) const;
	char& operator[](const unsigned int&);

	bool operator==(const GString&) const;
	bool operator!=(const GString&) const;

	bool operator==(const GType&) const;
	bool operator!=(const GType&) const;

	bool operator==(const char*) const;
	bool operator!=(const char*) const;

	GString operator+=(const char&);
	GString operator+=(const GType&);
	GString operator+=(const GString&);
	GString operator+=(const char*);

	bool operator<(const GString&) const;
	bool operator<=(const GString&) const;

	bool operator<(const GType&) const;
	bool operator<=(const GType&) const;

	bool operator<(const char*) const;
	bool operator<=(const char*) const;

	bool operator>(const GString&) const;
	bool operator>=(const GString&) const;

	bool operator>(const GType&) const;
	bool operator>=(const GType&) const;

	bool operator>(const char*) const;
	bool operator>=(const char*) const;

	// Strng Helpers
	void initEmpty();
	GString substr(unsigned int, unsigned int=0) const;

	// Member helpers
	static GType Typify(const char*, unsigned int);
	bool isWhitespace() const;
	bool isInteger() const;
	bool isFloat() const;
	bool isUpper() const;
	GString toUpper() const;
	bool isLower() const;
	GString toLower() const;
	GString toggleCase() const;
	GString trim() const;
	GString Stringify() const;

	// Static helpers
	static bool isWhitespace(char);
	static bool isWhitespace(const char*);
	static bool isWhitespace(const char*, unsigned int);
	static bool isWhitespace(const GString&);
	static bool isInteger(const char*);
	static bool isInteger(const char*, unsigned int);
	static bool isInteger(const GString&);
	static bool isFloat(const char*);
	static bool isFloat(const char*, unsigned int);
	static bool isFloat(const GString&);
	static bool isUpper(char);
	static bool isUpper(const char*);
	static bool isUpper(const char*, unsigned int);
	static bool isUpper(const GString&);
	static char toUpper(char);
	static GString toUpper(const char*);
	static GString toUpper(const char*, unsigned int);
	static GString toUpper(const GString&);
	static bool isLower(char);
	static bool isLower(const char*);
	static bool isLower(const char*, unsigned int);
	static bool isLower(const GString&);
	static char toLower(char);
	static GString toLower(const char*);
	static GString toLower(const char*, unsigned int);
	static GString toLower(const GString&);
	static char toggleCase(char);
	static GString toggleCase(const char*);
	static GString toggleCase(const char*, unsigned int);
	static GString toggleCase(const GString&);
	static GString trim(const char*);
	static GString trim(const char*, unsigned int);
	static GString trim(const GString&);

	static GString charTOstring(char);
	static GString shortTOstring(short);
	static GString intTOstring(int);
	static GString longTOstring(int64_t);
	static GString floatTOstring(float);
	static GString doubleTOstring(double);
	static GString datetimeTOstring(int64_t);
	static GString dateTOstring(int64_t);
	static GString timeTOstring(int64_t);
	static int64_t parseDate(const std::string, const std::string, const std::string,
		const std::string = "00", const std::string = "00", const std::string = "00");
};
};

shmea::GString operator+ (const shmea::GString& lhs, const shmea::GString& rhs);
shmea::GString operator+ (const shmea::GString& lhs, const char* rhs);
shmea::GString operator+ (const shmea::GString& lhs, char rhs);
shmea::GString operator+ (const char* lhs, const shmea::GString& rhs);
shmea::GString operator+ (char lhs, const shmea::GString& rhs);


#endif
