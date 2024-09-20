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
#ifndef _GCANDLE
#define _GCANDLE

#include <float.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

class Candle
{
private:
	float x, open, close, high, low;

public:
	Candle();
	Candle(float, float, float, float, float);
	Candle(const Candle&);
	virtual ~Candle();

	// get
	float getX() const;
	float getOpen() const;
	float getClose() const;
	float getHigh() const;
	float getLow() const;

	// set
	void set(float, float, float, float, float);
	void setX(float);
	void setOpen(float newOpen);
	void setClose(float newClose);
	void setHigh(float newHigh);
	void setLow(float newLow);

	// operators
	inline Candle operator+(Candle c)
	{
		c.x += x;
		c.open += open;
		c.close += close;
		c.high += high;
		c.low += low;
		return c;
	}

	inline Candle operator-(Candle c)
	{
		c.x = x - c.x;
		c.open = open - c.open;
		c.close = close - c.close;
		c.high = high - c.high;
		c.low = low - c.low;
		return c;
	}

	inline Candle operator*(float scalar)
	{
		return Candle(scalar * x, scalar * open, scalar * close, scalar * high, scalar * low);
	}

	inline Candle operator/(float scalar)
	{
		return Candle(x / scalar, open / scalar, close / scalar, high / scalar, low / scalar);
	}

	void operator=(const Candle& c)
	{
		setX(c.x);
		setOpen(c.open);
		setClose(c.close);
		setHigh(c.high);
		setLow(c.low);
	}

	bool operator==(const Candle& c) const
	{
		// X
		float deltaX = c.x - x;
		bool xChange = ((deltaX > -DBL_EPSILON) && (deltaX < DBL_EPSILON));
		// Open
		float deltaOpen = c.open - open;
		bool openChange = ((deltaOpen > -DBL_EPSILON) && (deltaOpen < DBL_EPSILON));
		// Close
		float deltaClose = c.close - close;
		bool closeChange = ((deltaClose > -DBL_EPSILON) && (deltaClose < DBL_EPSILON));
		// High
		float deltaHigh = c.high - high;
		bool highChange = ((deltaHigh > -DBL_EPSILON) && (deltaHigh < DBL_EPSILON));
		// Low
		float deltaLow = c.low - low;
		bool lowChange = ((deltaLow > -DBL_EPSILON) && (deltaLow < DBL_EPSILON));

		return (xChange && openChange && closeChange && highChange && lowChange);
	}

	bool operator<(const Candle& c2) const // ONLY COMPARES X
	{
		return (getX() < c2.getX());
	}

	bool operator>(const Candle& c2) const // ONLY COMPARES X
	{
		return (getX() > c2.getX());
	}
};

#endif
