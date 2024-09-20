//PNGPlotter.h
#ifndef PNGPLOTTER_H
#define PNGPLOTTER_H

#include "image.h"
#include "Candle.h"
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>
#include "standardizable.h"

namespace shmea{

class PNGPlotter : public GStandardizable
{
	private:
		Image image;
		unsigned width;
		unsigned height;
		long min_time;
		long max_time;
		float min_price;
		float max_price;

		RGBA color_bullish;
		RGBA color_bearish;

		std::vector<Candle> dataPoints;

		void drawCandleStick(Image&, int, int, int, int, int, RGBA&);
		void redraw();
	
	public:
		PNGPlotter(unsigned, unsigned);
		void addDataPoint(long, float, float, float, float);
		void SavePNG(const std::string&);
	
};
};
#endif
