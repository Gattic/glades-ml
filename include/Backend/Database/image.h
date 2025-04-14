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
#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector> 
#include "GVector.h"

class RUBackgroundComponent;

namespace shmea {

class GString;
class GList;

struct RGBA {
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
	
	RGBA() : r(0), g(0), b(0), a(0) {}
	RGBA(unsigned char r_, unsigned char g_, unsigned char b_, unsigned char a_)
		: r(r_), g(g_), b(b_), a(a_) {}
};

//	bmp, png	(when T == RGBA)
//	ppm			(when T == RGB)
//	pbm			(when T == bool)

class Image
{
	friend RUBackgroundComponent;
	friend class PNGHelper;

protected:
	unsigned int width;
	unsigned int height;
	int pitch;
	RGBA* data;

	void copy_helper(const Image& image)
	{
		pitch = image.pitch;
		Allocate(image.getWidth(), image.getHeight());
		for (unsigned int i = 0; i < image.getWidth(); i++)
		{
			for (unsigned int j = 0; j < image.getHeight(); j++)
				this->SetPixel(i, j, image.GetPixel(i, j));
		}
	}

public:

	Image() : width(0), height(0), pitch(0), data(NULL)
	{
		//
	}

	Image(const Image& image) : data(NULL)
	{
		copy_helper(image);
	}

	const Image& operator=(const Image& image)
	{
		if (this != &image)
			copy_helper(image);
		return *this;
	}

	virtual ~Image()
	{
		delete[] data;
	}

	// initialize an image of a specific size
	void Allocate(unsigned int width, unsigned int height)
	{
		this->width = width;
		this->height = height;
		delete[] data;
		if (width == 0 && height == 0)
		{
			data = NULL;
		}
		else
		{
			if (!(width > 0 && height > 0))
				return;
			data = new RGBA[width * height];
		}
	}

	// gets
	unsigned int getWidth() const
	{
		return width;
	}

	unsigned int getHeight() const
	{
		return height;
	}

	unsigned int getPixelCount() const
	{
		return width*height;
	}

	std::vector<unsigned char> getPixels() const;

	void drawVerticalGradient(int x, int y, const RGBA& topColor, const RGBA& bottomColor, int cornerRadius = 0);
	void drawHorizontalGradient(int x, int y, const RGBA& leftColor, const RGBA& rightColor, int cornerRadius = 0);

	RGBA averageColor(int startX, int startY, int width, int height) const;

	RGBA GetPixel(unsigned int x, unsigned int y) const
	{
		if (!(x < width))
			return RGBA();

		if (!(y < height))
			return RGBA();

		return data[y * width + x];
	}

	// sets
	void SetAllPixels(const RGBA& value)
	{
		for (unsigned int i = 0; i < width * height; i++)
			data[i] = value;
	}

	void SetPixel(unsigned int x, unsigned int y, const RGBA& value)
	{
		if (!(x < width))
			return;

		if (!(y < height))
			return;

		data[y * width + x] = value;
	}

	bool LoadPPM(const GString&);
	bool LoadPBM(const GString&);
	bool SavePPM(const GString&) const;
	bool SavePBM(const GString&) const;
	void LoadBMP(const GString&);
	void SavePNG(const GString&) const;
	void LoadPNG(const GString&);

	shmea::GVector<float> flatten() const;
	float rgbaToHueIntensity(const RGBA&) const;

	shmea::GString hash() const;
	bool operator<(const Image&) const;
	bool operator>(const Image&) const;

};
};

#endif
