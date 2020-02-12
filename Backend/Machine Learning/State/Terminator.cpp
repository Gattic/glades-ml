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
#include "Terminator.h"

glades::Terminator::Terminator()
{
	timestamp = 0l;
	epoch = 0l;
	accuracy = 0.0f;
}

glades::Terminator::~Terminator()
{
	timestamp = 0l;
	epoch = 0l;
	accuracy = 0.0f;
}

int64_t glades::Terminator::getTimestamp() const
{
	return timestamp;
}

int64_t glades::Terminator::getEpoch() const
{
	return epoch;
}

float glades::Terminator::getAccuracy() const
{
	return accuracy;
}

void glades::Terminator::setTimestamp(int64_t newTimstamp)
{
	timestamp = newTimstamp;
}

void glades::Terminator::setEpoch(int64_t newEpoch)
{
	epoch = newEpoch;
}

void glades::Terminator::setAccuracy(float newAccuracy)
{
	accuracy = newAccuracy;
}

bool glades::Terminator::triggered(int64_t cTimestamp, int64_t cEpoch, float cAccuracy) const
{
	// printf("CURRENT(%ld,%ld,%f)\n", cTimestamp, cEpoch, cAccuracy);
	// printf("LIMIT(%ld,%ld,%f)\n", timestamp, epoch, accuracy);
	if ((timestamp > 0l) && (cTimestamp % timestamp == 0))
		return true;

	if ((epoch > 0l) && (cEpoch % epoch == 0))
		return true;

	if ((accuracy > 0.0f) && (cAccuracy >= accuracy))
		return true;

	// no conditions triggered
	return false;
}
