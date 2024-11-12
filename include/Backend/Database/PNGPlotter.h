//PNGPlotter.h
#ifndef PNGPLOTTER_H
#define PNGPLOTTER_H

#include "image.h"
#include <string>
#include <limits>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace shmea{

class PNGPlotter
{

	private:
		Image image;
		unsigned int width;
		unsigned int height;
		float min_price, max_price;
		const int margin_top;
		const int margin_right;
		const int margin_bottom;
		const int margin_left;
		long last_timestamp;
		int total_candles_drawn;
		const int max_candles;
		int candle_width;
		int last_candle_pos;
		int lines;
		std::vector<bool> first_line_point;
		std::vector<int> last_price_pos;
		int last_line_drawn;
		std::vector<RGBA> line_colors;
		std::map<std::string, RGBA> indicatorColors;
		std::vector<std::string> line_color_names;
		RGBA color_bullish;
		RGBA color_bearish;


//		RGB HSLToRGB(float, float, float);
//		void generateUniqueColors(int);
		void initialize_colors(std::vector<RGBA>&, std::vector<std::string>&);
		Image downsampleToTargetSize();

		void drawGrid(int = 7, int = 16);
			
		void drawLine(int, int, int, int, RGBA&);
		void drawCandleStick(Image&, int, int, int, int, int, RGBA&);
	public:

		

		static const int TARGET_WIDTH = 1800;
		static const int TARGET_HEIGHT = 1200;
		static const int SUPERSAMPLE_SCALE = 4;
		static const int SUPERSAMPLE_WIDTH = TARGET_WIDTH * SUPERSAMPLE_SCALE;
		static const int SUPERSAMPLE_HEIGHT = TARGET_HEIGHT * SUPERSAMPLE_SCALE;
		
		PNGPlotter(unsigned int, unsigned int, int, double, double, int = 0, int=0, int=0, int=0, int=0);
		void addDataPointWithIndicator(double, int = 0, std::string = "", std::string = "");
		void addDataPoint(double, int = 0, bool = true, RGBA* = NULL);
		void drawNewCandle(long, float, float, float, float);
		void SavePNG(const std::string&, const std::string&);
	
};
};
#endif
