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
#ifndef _GLOGGER
#define _GLOGGER

#include "GLogger.h"
#include "GList.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class Serializable;

class GLogger
{
	friend Serializable;

private:
	// Keys are categories
	int printLevel;
	shmea::GList infoKeys;
	shmea::GList verboseKeys;
	shmea::GList debugKeys;
	shmea::GList warningKeys;
	shmea::GList errorKeys;
	shmea::GList fatalKeys;
	// The above and below lists are aligned
	shmea::GList infoLog;
	shmea::GList verboseLog;
	shmea::GList debugLog;
	shmea::GList warningLog;
	shmea::GList errorLog;
	shmea::GList fatalLog;

	// override variables to disable specific output
	bool surpressInfo;
	bool surpressVerbose;
	bool surpressDebug;
	bool surpressWarning;
	bool surpressError;
	bool surpressFatal;

public:

	const static int LOG_NONE = -1;
	const static int LOG_INFO = 0;
	const static int LOG_VERBOSE = 1;
	const static int LOG_DEBUG = 2;
	const static int LOG_WARNING = 3;
	const static int LOG_ERROR = 4;
	const static int LOG_FATAL = 5;

	GLogger();
	GLogger(int);
	GLogger(const GLogger&);
	virtual ~GLogger();

	void copy(const GLogger&);
	void clear();
	void setPrintLevel(int);
	int getPrintLevel() const;
	void surpress(int);
	void unsurpress(int);
	bool surpressCheck(int) const;

	void log(int, shmea::GString, shmea::GString);
	void info(shmea::GString, shmea::GString);
	void verbose(shmea::GString, shmea::GString);
	void debug(shmea::GString, shmea::GString);
	void warning(shmea::GString, shmea::GString);
	void error(shmea::GString, shmea::GString);
	void fatal(shmea::GString, shmea::GString);

	//void print() const;
};
};

#endif
