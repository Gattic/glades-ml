//PNGPlotter.h
#ifndef PNGPLOTTER_H
#define PNGPLOTTER_H

#include "image.h"
#include <ft2build.h>
#include FT_FREETYPE_H
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
		const bool fourQuadrants;
		long last_timestamp;
		int total_candles_drawn;
		const int graphSize;
		int candle_width;
		int last_candle_pos;
		int lines;
		std::vector<bool> first_line_point;
		std::vector<int> last_price_pos;
		int last_line_drawn;
		std::vector<RGBA> line_colors;
		std::map<std::string, RGBA> indicatorColors;
		std::map<std::string, RGBA> indicatorTextColor;
		std::map<std::string, int> indicatorPoint;
		std::vector<std::string> line_color_names;
		RGBA color_bullish;
		RGBA color_bearish;

		unsigned int headerPenXStarting;
		unsigned int headerPenYStarting;
		unsigned int headerXSpacing;
		unsigned int headerYSpacing;

		std::vector<std::vector<unsigned int> > headerSpacings;

		std::map<int, std::string> AGG_SIZE;
		//font
		FT_Library ft;
		FT_Face face;
		std::vector<float> horizontalLabels;


//		RGB HSLToRGB(float, float, float);
//		void generateUniqueColors(int);
		void initialize_colors(std::vector<RGBA>&, std::vector<std::string>&);
		void initialize_font(const std::string = "fonts/font.ttf");
		Image downsampleToTargetSize();

		void drawFourQuadrants();	

		void drawPoint(int, int, int, RGBA&);
		void drawLine(int, int, int, int, RGBA&, int = 6);
		void drawCandleStick(Image&, int, int, int, int, int, RGBA&);
		void drawArrow(int, int, int, int, RGBA&, int);
		void drawHistogram(int, int, int, RGBA&);
	public:

		
		static const int TARGET_WIDTH = 2400;
		static const int TARGET_HEIGHT = 1200;
		static const int SUPERSAMPLE_SCALE = 4;
		static const int SUPERSAMPLE_WIDTH = TARGET_WIDTH * SUPERSAMPLE_SCALE;
		static const int SUPERSAMPLE_HEIGHT = TARGET_HEIGHT * SUPERSAMPLE_SCALE;
		
		PNGPlotter(unsigned int, unsigned int, int, double, double, int = 0, int=0, int=0, int=0, int=0, bool = false);
		void addDataPointWithIndicator(double, int = 0, std::string = "", std::string = "");
		void addDataPoint(double, int = 0, bool = true, RGBA* = NULL, int = 6);
		void addDataPointsPCA(std::vector<std::vector<double> >&, RGBA&);
		void addArrow(std::vector<std::vector<double> >&, RGBA&, int = 10);
		void addHistogram(std::vector<int>&, RGBA&);
		void drawNewCandle(long, float, float, float, float);
		void SavePNG(const std::string&, const std::string&);

		int getWidth();
		int getHeight();
	
		void drawYGrid();
		void drawXGrid(int64_t, int64_t);

		void HeaderPNG(const std::string&, unsigned int, unsigned int = 0, unsigned int = 0, RGBA = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
		void GraphLabel(unsigned int, unsigned int, const std::string&, unsigned int, unsigned int=0, unsigned int=0, bool = false, RGBA = RGBA(0xFF, 0xFF, 0xFF, 0xFF), RGBA = RGBA(0xFF, 0xFF, 0xFF, 0xFF) );

		std::string aggString(int);
};
};
#endif
