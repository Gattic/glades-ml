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
#ifndef _GIMAGE_H_
#define _GIMAGE_H_

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector> 

class RUBackgroundComponent;

namespace shmea {

class GString;
class GList;

// 32 bit color pixel density
class RGBA
{
public:
	RGBA(unsigned char red = 0xFF, unsigned char green = 0xFF, unsigned char blue = 0xFF,
		 unsigned char alpha = 0xFF)
		: r(red), g(green), b(blue), a(alpha)
	{
		//
	}

	bool isWhite() const
	{
		return r == 0xFF && g == 0xFF && b == 0xFF;
	}

	unsigned char r, g, b, a;
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
	void Allocate(int w, int h)
	{
		width = w;
		height = h;
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

	void drawVerticalGradient(int, int, RGBA, RGBA, int);

	RGBA averageColor(int, int, int, int);

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

	shmea::GList flatten() const;
	bool unflatten(const shmea::GList&);

	shmea::GString hash() const;
	bool operator<(const Image&) const;
	bool operator>(const Image&) const;

};
};

#endif
