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
#ifndef _GSERIALIZABLE
#define _GSERIALIZABLE

#include "GList.h"
#include "gtable.h"
#include <algorithm>
#include <dirent.h>
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/signal.h>
#include <sys/stat.h>
#include <vector>

namespace shmea {
class Serializable
{
private:
	static const std::string NEED_ESCAPING;
	static const char ESC_CHAR;

	static bool escapeSeparators(char**, unsigned int&);
	static bool addDelimiter(char**, unsigned int&, bool);
	static bool addItemToSerial(char**, unsigned int&, char*, int, int, int);
	static bool serializeItem(const GType&, bool, char**, unsigned int&);

	static bool isDelimiterAt(const char*, int, const char*);
	static int findNextDelimiterIndex(const char*, int, const char*, const int, const int);
	static bool isEscaped(const int, const char*);
	static int getDelimiterIdx(const char*, const int, const char*, const int, bool);

	static int deserializeType(char**, int&, int64_t&);
	static int64_t deserializeSize(char**, int&, int64_t&);
	static int unescapeCharacter(char**, int64_t&, int);
	static char* deserializeContent(char**, int&, int64_t&, int64_t, int);

	// virtual GTable toGTable()const = 0;
	// virtual void fromGTable(const GTable&)=0;

public:
	static int Serialize(const GList&, char**); // to byte stream
	// static int Serialize(const GTable&, char**);	   // to byte stream
	static GList DeserializeHelper(const char*, int); // from byte stream; make this private
	// static GTable Deserialize(const char*, int);	  // from byte stream
	GTable toGTable() const;
};
};

#endif
