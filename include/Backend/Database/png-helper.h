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
#ifndef _PNG_HELPER
#define _PNG_HELPER

#include <iostream>
#include <cstring>
#include <cmath>
#include <png.h>

namespace shmea
{

class Image;

class PNGHelper
{

public:
    static void applyRainbowFilter(png_bytep, png_bytep, png_uint_32, png_byte);

    static void readImage(const char* inputPath, png_structp& png, png_infop& info, png_bytep*& rowPointers, png_uint_32& width, png_uint_32& height, png_byte& bitDepth, png_byte& colorType);
    static void writeImage(const char* outputPath, png_structp png, png_infop info, png_bytep* rowPointers, png_uint_32 width, png_uint_32 height, png_byte bitDepth, png_byte colorType);

    static void pngTest(const char*, const char*);
    static void LoadPNG(Image&, const char*);

};

};

#endif
