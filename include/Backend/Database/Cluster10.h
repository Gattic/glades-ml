// Cluster10.h
#ifndef CLUSTER10_H
#define CLUSTER10_H

#include "image.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <string>
#include <vector>
#include <map>
#include <cmath>

namespace shmea {

// Forward declarations
class ThemeManager;
class GridRenderer;
class TextRenderer;

class Cluster10 {
public:
    // Data structures
    struct Point {
        double x;
        double y;
        Point() : x(0.0), y(0.0) {}
        Point(double x_, double y_) : x(x_), y(y_) {}
    };

    struct CandleData {
        double timestamp;  // Unix timestamp for the candle
        double open;
        double close;
        double high;
        double low;
        
        CandleData() : timestamp(0.0), open(0.0), close(0.0), high(0.0), low(0.0) {}
        CandleData(double t, double o, double c, double h, double l) 
            : timestamp(t), open(o), close(c), high(h), low(l) {}
    };

    struct AxisRange {
        double min;
        double max;
        double padding;
        
        AxisRange() : min(0.0), max(1.0), padding(0.05) {}
        AxisRange(double min_, double max_, double padding_) 
            : min(min_), max(max_), padding(padding_) {}
    };

    // Constants for standard sizes
    static const int DEFAULT_WIDTH = 2400;
    static const int DEFAULT_HEIGHT = 1200;
    
    // Constructor/Destructor
    Cluster10(unsigned int width = DEFAULT_WIDTH, unsigned int height = DEFAULT_HEIGHT, 
              unsigned int margin_top = 60, unsigned int margin_right = 60, 
              unsigned int margin_bottom = 60, unsigned int margin_left = 60);
    ~Cluster10();
    
    // Basic controls
    void setShowGrid(bool show);
    void setShowAxes(bool show);
    void setCornerRadius(int radius);
    
    // Chart visualization methods
    void plotPoints(const std::vector<Point>& points, const RGBA& color, int pointSize = 8);
    void plotLine(const std::vector<Point>& points, const RGBA& color, int lineWidth = 2);
    void plotHistogram(const std::vector<int>& bins, const RGBA& color);
    void plotCandlestickChart(const std::vector<CandleData>& candles, 
                             const RGBA& bullishColor = RGBA(0x03, 0xC0, 0x3C, 0xFF),
                             const RGBA& bearishColor = RGBA(0xFF, 0x47, 0x45, 0xFF));
    void plotClusters(const std::vector<std::vector<double> >& data, const std::vector<int>& labels, 
                     const std::vector<std::vector<double> >& centroids = std::vector<std::vector<double> >());
    
    // Text and label methods
    void addTitle(const std::string& text, unsigned int fontSize = 36);
    void addAxisLabels(const std::string& xLabel, const std::string& yLabel, unsigned int fontSize = 24);
    void addLegend(const std::vector<std::string>& labels, const std::vector<RGBA>& colors, 
                  int x, int y, unsigned int fontSize = 18);
    
    // Save methods
    void saveAsPNG(const std::string& filename, const std::string& folder = ".");

private:
    // Internal state
    Image image;
    unsigned int width;
    unsigned int height;
    unsigned int margin_top;
    unsigned int margin_right;
    unsigned int margin_bottom;
    unsigned int margin_left;
    
    // Theme and color management
    std::vector<RGBA> themeColors;
    std::map<std::string, RGBA> elementColors;
    
    // Font handling
    FT_Library ft;
    FT_Face face;
    
    // Design elements settings
    bool showGrid;
    bool showAxes;
    int cornerRadius;
    
    // Initialization methods
    void initialize_colors();
    void initialize_font(const std::string fontPath = "fonts/font.ttf");
    
    // Background and layout methods
    void drawBackground();
    void drawGrid();
    void drawAxes();
    
    // Text rendering
    void drawText(int x, int y, const std::string& text, const RGBA& color, 
                 unsigned int fontSize = 18, bool centerAligned = false);
    void drawVerticalText(const std::string& text, int x, int y, int fontSize, const RGBA& color);
    
    // Primitive drawing methods
    void drawPoint(int x, int y, int size, const RGBA& color);
    void drawLine(int x1, int y1, int x2, int y2, const RGBA& color, int width = 2);
    void drawRect(int x, int y, int width, int height, const RGBA& color, bool filled = true, int borderWidth = 1);
    void drawCircle(int x, int y, int radius, const RGBA& color, bool filled = true, int borderWidth = 1);
    void drawCircle(Image& targetImage, int x, int y, int radius, const RGBA& color, bool filled = true, int borderWidth = 1);
    void drawCornerRadius(int x, int y, int radius, bool topLeft, bool topRight, bool bottomRight, bool bottomLeft);
    
    // Helper methods
    inline int clamp(int value, int min, int max);
    
    // Chart-specific components
    void drawCandlestick(int x, int y_open, int y_close, int y_high, int y_low, const RGBA& color);
    
    // Histogram-specific helpers
    void drawHistogramYAxis(int maxValue, int numTicks);
    void drawHistogramBars(const std::vector<int>& bins, int maxBinValue, int totalBars, 
                          int barWidth, int barSpacing, const RGBA& color);
    void drawHistogramBarHighlights(int x, int y, int barWidth, int barHeight);
    void drawHistogramStats(const std::vector<int>& bins, int maxBinValue);
    
    // Candlestick chart helpers
    void drawCandlestickXAxis(const std::vector<CandleData>& candles, int maxVisibleCandles, 
                             int totalCandles, double firstTimestamp, double lastTimestamp);
    void drawCandlestickYAxis(double minPrice, double maxPrice, int numTicks);
    void drawCandlestickPriceInfo(const std::vector<CandleData>& candles,
                                const RGBA& bullishColor, const RGBA& bearishColor);
    
    // Data scaling helpers
    AxisRange calculateXRange(const std::vector<Point>& points);
    AxisRange calculateYRange(const std::vector<Point>& points);
    AxisRange calculateXRange(const std::vector<std::vector<double> >& data);
    AxisRange calculateYRange(const std::vector<std::vector<double> >& data);
    
    // Coordinate mapping
    Point mapDataToScreen(double x, double y, const AxisRange& xRange, const AxisRange& yRange);
    
    // Layout calculation
    int getPlotWidth() const { return width - margin_left - margin_right; }
    int getPlotHeight() const { return height - margin_top - margin_bottom; }
};

} // namespace shmea

#endif // CLUSTER10_H 