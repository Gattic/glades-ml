#ifndef PNGPLOTTER_H
#define PNGPLOTTER_H

#include "../Database/image.h"
#include "Drawers/BaseDrawer.h"
#include "Drawers/GridDrawer.h"
#include "Drawers/CandlestickDrawer.h"
#include "Drawers/ArrowDrawer.h"
#include "Drawers/HistogramDrawer.h"
#include "Drawers/LabelsDrawer.h"
#include "Drawers/DataDrawer.h"
#include "Drawers/LegendDrawer.h"

#include <string>
#include <limits>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>

namespace shmea {

class PNGPlotter {
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
    
    // Specialized drawer components
    GridDrawer* gridDrawer;
    CandlestickDrawer* candlestickDrawer;
    ArrowDrawer* arrowDrawer;
    HistogramDrawer* histogramDrawer;
    LabelsDrawer* labelsDrawer;
    DataDrawer* dataDrawer;
    LegendDrawer* legendDrawer;
    
    // Standard colors
    RGBA color_bullish;
    RGBA color_bearish;

    // Aggregation time mappings
    std::map<int, std::string> AGG_SIZE;

    // Downsample the high-res image to target size
    Image downsampleToTargetSize();

    // Title and axis labels
    std::string title;
    std::string xAxisLabel;
    std::string yAxisLabel;
    RGBA titleColor;
    RGBA xAxisLabelColor;
    RGBA yAxisLabelColor;
    unsigned int titleFontSize;
    unsigned int axisLabelFontSize;

public:
    static const int TARGET_WIDTH = 2400;
    static const int TARGET_HEIGHT = 1200;
    static const int SUPERSAMPLE_SCALE = 4;
    static const int SUPERSAMPLE_WIDTH = TARGET_WIDTH * SUPERSAMPLE_SCALE;
    static const int SUPERSAMPLE_HEIGHT = TARGET_HEIGHT * SUPERSAMPLE_SCALE;
    
    /**
     * Creates a new PNGPlotter with the specified dimensions and settings.
     * 
     * @param width Width of the image in pixels
     * @param height Height of the image in pixels
     * @param graphSize Number of data points that can fit horizontally
     * @param options Optional parameters in a key-value map:
     *        - "max_price": Maximum price value (default: 100.0)
     *        - "min_price": Minimum price value (default: 0.0)
     *        - "lines": Number of lines to plot (default: 0)
     *        - "margin_top": Top margin in pixels (default: height*0.1)
     *        - "margin_right": Right margin in pixels (default: width*0.1)
     *        - "margin_bottom": Bottom margin in pixels (default: height*0.15)
     *        - "margin_left": Left margin in pixels (default: width*0.15)
     *        - "four_quadrants": Whether to draw four quadrants (default: false)
     *        - "title": Chart title (default: "Data Visualization")
     *        - "x_axis_label": X-axis label (default: "Time")
     *        - "y_axis_label": Y-axis label (default: "Value")
     */
    PNGPlotter(unsigned int width, unsigned int height, int graphSize, 
               const std::map<std::string, std::string>& options = std::map<std::string, std::string>());
    
    // Legacy constructor (kept for backward compatibility)
    PNGPlotter(unsigned int width, unsigned int height, int graphSize, 
               double max_price, double low_price, int lines = 0, 
               int margin_top = 0, int margin_right = 0, 
               int margin_bottom = 0, int margin_left = 0, bool fourQuadrants = false);
    
    ~PNGPlotter();
    
    // Data point addition methods
    void addDataPointWithIndicator(double newPrice, int portIndex = 0, std::string indicator = "", std::string value = "");
    void addDataPoint(double newPrice, int portIndex = 0, bool draw = true, RGBA* lineColor = NULL, int lineWidth = 6);
    void addDataPointsPCA(const std::vector<std::vector<double> >& data, const RGBA& pointColor);
    void addDataPointsKMeans(const std::string& graphName, 
                             const std::vector<std::vector<double> >& data, 
                             const std::vector<int>& labels, 
                             const std::vector<std::vector<float> >& centroids);
    void addArrow(const std::vector<std::vector<double> >& sorted_eig_vecs, 
                  const std::vector<double>& variance_explained, 
                  const RGBA& arrowColor);
    void addHistogram(const std::vector<int>& bins, RGBA& barColor);
    
    // Candlestick charting
    void drawNewCandle(long timestamp, float open, float close, float high, float low);
    
    // Grid and labels
    void drawYGrid();
    void drawXGrid(int64_t start, int64_t end);
    // New method to draw grid with labels together
    void drawGridWithLabels(int64_t start, int64_t end);
    void HeaderPNG(const std::string& text, unsigned int fontSize, 
                   unsigned int headerPos = 0, unsigned int rePositionY = 0, 
                   RGBA headerTextColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    void GraphLabel(unsigned int penX, unsigned int penY, 
                    const std::string& text, unsigned int fontSize, 
                    unsigned int xOffset = 0, unsigned int yOffset = 0, 
                    bool hasBox = false, 
                    RGBA labelColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF), 
                    RGBA textColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    
    // Title and axis labels methods
    void setTitle(const std::string& title, unsigned int fontSize = 300, 
                 RGBA titleColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    void setXAxisLabel(const std::string& label, unsigned int fontSize = 200, 
                      RGBA labelColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    void setYAxisLabel(const std::string& label, unsigned int fontSize = 200, 
                      RGBA labelColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    
    // Getters for titles and labels
    const std::string& getTitle() const { return title; }
    const std::string& getXAxisLabel() const { return xAxisLabel; }
    const std::string& getYAxisLabel() const { return yAxisLabel; }
    
    // Legend methods
    void addLegendEntry(const std::string& label, const RGBA& color);
    void drawLegend();
    
    // File operations
    void SavePNG(const std::string& filename, const std::string& folder);
    
    // Utility functions
    int getWidth();
    int getHeight();
    std::string aggString(int aggSize);
};

} // namespace shmea
#endif
