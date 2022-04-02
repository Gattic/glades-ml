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
#include "GTable.h"
#include "GObject.h"
#include "GString.h"
#include "ServiceData.h"
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
	static const GString NEED_ESCAPING;
	static const char ESC_CHAR;

	static GString escapeSeparators(const GType&);
	static GString addDelimiter(const GString&, bool);
	static GString addItemToSerial(int, unsigned int, const GType&);
	static GString serializeItem(const GType&, bool);

	static bool isDelimiterAt(const char*, int, const char*);
	static int findNextDelimiterIndex(int, const GString&, const GString&);
	static bool isEscaped(const int, const char*);
	static int getDelimiterIdx(const GString&, const GString&, bool);

	static GString deserializeContent(const GString&);

public:

	static GString Serialize(const GList&, bool = false);		// to byte stream
	static GString Serialize(const GTable&, bool = false);		// to byte stream
	static GString Serialize(const GObject&, bool = false);		// to byte stream
	static GString Serialize(const ServiceData*);			// to byte stream

	static int Deserialize(GList&, const GString& serial, int = 0);// from byte stream; make this private?
	static void Deserialize(GTable&, const GString& serial);	// from byte stream
	static void Deserialize(GObject&, const GString& serial);	// from byte stream
	static void Deserialize(ServiceData*, const GString& serial);	// from byte stream

	virtual GObject serialize() const = 0;
	virtual void deserialize(const GObject&) = 0;

};
};

#endif
