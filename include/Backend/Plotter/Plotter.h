#ifndef PLOTTER_H
#define PLOTTER_H

#include "../Database/image.h"
#include "DataMapper.h"
#include <map>
#include <string>
#include <vector>
#include <ft2build.h>
#include FT_FREETYPE_H

// Forward declarations to avoid circular dependencies
namespace shmea {
    class ColorManager;
    class SuperSamplingManager;
    class ChartLayout;
    class ShapeRenderer;
    class TextRenderer;
    class GridRenderer;
    class DataMapper;
    class ChartStyler;
    
    // Forward declare Plotter for ChartBuilder
    class Plotter;
}

namespace shmea {

// Chart types for margin calculation
enum ChartType {
    CHART_DEFAULT,
    CHART_HISTOGRAM, 
    CHART_LINE,
    CHART_SCATTER,
    CHART_CANDLESTICK,
    CHART_CLUSTER
};

// Series type for multi-series charts
enum SeriesType {
    SERIES_LINE,
    SERIES_SCATTER,
    SERIES_AREA
};

// Data point structure for visualizations
struct Point {
    double x;
    double y;
    
    Point() : x(0), y(0) {}
    Point(double x_, double y_) : x(x_), y(y_) {}
};

// Arrow structure for visualization
struct Arrow {
    Point start;
    Point end;
    RGBA color;
    int lineWidth;
    int arrowheadSize;
    
    Arrow() : lineWidth(2), arrowheadSize(10) {}
    
    Arrow(const Point& start_, const Point& end_, const RGBA& color_ = RGBA(),
          int lineWidth_ = 2, int arrowheadSize_ = 10)
        : start(start_), end(end_), color(color_), 
          lineWidth(lineWidth_), arrowheadSize(arrowheadSize_) {}
    
    Arrow(double startX, double startY, double endX, double endY, 
          const RGBA& color_ = RGBA(), int lineWidth_ = 2, int arrowheadSize_ = 10)
        : start(startX, startY), end(endX, endY), color(color_), 
          lineWidth(lineWidth_), arrowheadSize(arrowheadSize_) {}
};

// Candlestick data structure
struct CandleData {
    double timestamp;
    double open;
    double close;
    double high;
    double low;
    
    CandleData() : timestamp(0), open(0), close(0), high(0), low(0) {}
    CandleData(double t, double o, double c, double h, double l)
        : timestamp(t), open(o), close(c), high(h), low(l) {}
};

// Struct for chart configuration - kept here for backward compatibility
struct ChartConfig {
    std::string title;
    unsigned int titleFontSize;
    std::string xAxisLabel;
    std::string yAxisLabel;
    unsigned int axisFontSize;
    
    ChartConfig(const std::string& title_, unsigned int titleFontSize_,
                const std::string& xAxisLabel_, const std::string& yAxisLabel_,
                unsigned int axisFontSize_)
        : title(title_), titleFontSize(titleFontSize_),
          xAxisLabel(xAxisLabel_), yAxisLabel(yAxisLabel_),
          axisFontSize(axisFontSize_) {}
};

// Series data for multi-series charts
struct Series {
    std::string name;
    std::vector<Point> data;
    RGBA color;
    SeriesType type;
    int lineWidth;   // Used for SERIES_LINE
    int pointSize;   // Used for SERIES_SCATTER
    
    Series() : type(SERIES_LINE), lineWidth(2), pointSize(8) {}
    
    Series(const std::string& name_, const std::vector<Point>& data_, 
           const RGBA& color_, SeriesType type_ = SERIES_LINE,
           int lineWidth_ = 2, int pointSize_ = 8)
        : name(name_), data(data_), color(color_), type(type_),
          lineWidth(lineWidth_), pointSize(pointSize_) {}
};

// ChartBuilder: A fluent API for building charts
class ChartBuilder {
public:
    ChartBuilder(Plotter& plotter);
    
    // Common setup methods
    ChartBuilder& title(const std::string& title, unsigned int fontSize = 36);
    ChartBuilder& size(unsigned int width, unsigned int height, unsigned int ssaaFactor = 1);
    ChartBuilder& margins(unsigned int top, unsigned int right, 
                          unsigned int bottom, unsigned int left);
    ChartBuilder& autoMargins(ChartType chartType = CHART_DEFAULT);
    ChartBuilder& axisLabels(const std::string& xLabel, const std::string& yLabel,
                            unsigned int fontSize = 28);
    ChartBuilder& grid(bool show);
    ChartBuilder& axes(bool show);
    ChartBuilder& originAxes(bool show);
    ChartBuilder& cornerRadius(int radius);
    ChartBuilder& logo(const std::string& logoPath);
    ChartBuilder& colors(const std::vector<RGBA>& colors);
    
    // Data visualization methods
    ChartBuilder& addSeries(const Series& series);
    ChartBuilder& addSeries(const std::string& name, const std::vector<Point>& data, 
                           const RGBA& color, SeriesType type = SERIES_LINE,
                           int lineWidth = 2, int pointSize = 8);
    ChartBuilder& addHistogramData(const std::vector<int>& bins, 
                                  const RGBA& color = RGBA(), 
                                  bool showXAxisLabels = true);
    ChartBuilder& addHistogramDataWithLabels(const std::vector<int>& bins, 
                                           const std::vector<std::string>& labels,
                                           const RGBA& color = RGBA());
    ChartBuilder& addCandlestickData(const std::vector<CandleData>& candles,
                                    const RGBA& bullishColor = RGBA(0x03, 0xC0, 0x3C, 0xFF),
                                    const RGBA& bearishColor = RGBA(0xFF, 0x47, 0x45, 0xFF));
    ChartBuilder& addClusterData(const std::vector<std::vector<double> >& data,
                                const std::vector<int>& labels,
                                const std::vector<std::vector<double> >& centroids);
    
    // Centroid positioning method
    ChartBuilder& alignCentroidsWithClusters(bool align = true);
    
    // Arrow visualization
    ChartBuilder& addArrows(const std::vector<Arrow>& arrows);
    ChartBuilder& addArrow(const Arrow& arrow);
    ChartBuilder& addArrow(double startX, double startY, double endX, double endY, 
                          const RGBA& color = RGBA(), int lineWidth = 2, int arrowheadSize = 10);
    
    // Rendering
    void saveAs(const std::string& filename, const std::string& folder = ".");
    
private:
    Plotter& plotter;
    ChartType chartType;
    std::vector<Series> series;
    std::vector<Arrow> arrows;
    bool hasHistogramData;
    std::vector<int> histogramBins;
    RGBA histogramColor;
    bool histogramShowXAxisLabels;
    bool hasLabeledHistogramData;
    std::vector<std::string> histogramLabels;
    bool hasCandlestickData;
    std::vector<CandleData> candlestickData;
    RGBA bullishColor;
    RGBA bearishColor;
    bool hasClusterData;
    std::vector<std::vector<double> > clusterData;
    std::vector<int> clusterLabels;
    std::vector<std::vector<double> > centroids;
    bool alignCentroids; // Whether to force centroids to align with cluster centers
};

// Plotter class that handles visualization using the component classes
class Plotter {
public:
    static std::map<unsigned int, std::string> AGG_SIZE;
    // Constructor and destructor
    Plotter(unsigned int width = 800, unsigned int height = 600, unsigned int ssaa_factor = 1);

    void initAggSize();

    void initialize_font(const std::string);
    
    ~Plotter();
    
    // Main entry point for fluent chart building API
    ChartBuilder chart();
    
    // Initialization
    void initialize();
    
    // State management for margin consistency
    void saveState();
    void restoreState();
    
    // Chart configuration methods
    void setShowGrid(bool show);
    void setShowAxes(bool show);
    void setShowOriginAxes(bool show);
    void drawOriginAxes(double xMin, double xMax, double yMin, double yMax);
    void setCornerRadius(int radius);
    void setSuperSamplingFactor(unsigned int factor);
    void setMarginTop(unsigned int margin);
    void setMarginRight(unsigned int margin);
    void setMarginBottom(unsigned int margin);
    void setMarginLeft(unsigned int margin);
    
    // Auto margin calculation
    void calculateOptimalMargins(ChartType chartType = CHART_DEFAULT);
    
    // Custom color manager
    void setCustomColors(const std::vector<RGBA>& clusterColors);
    void use10ClusterColorScheme(); // Use the 10-cluster color scheme
    
    // Basic drawing methods
    void prepareCanvas();
    void addTitle(const std::string& text, unsigned int fontSize = 24);
    void addAxisLabels(const std::string& xLabel, const std::string& yLabel, unsigned int fontSize = 18);
    int addLegend(const std::vector<std::string>& labels, const std::vector<RGBA>& colors,
                  int x, int y, unsigned int fontSize = 18, bool _new = false, bool redraw = false);
    void setYAxisLabel(const std::string& label);
    
    // Logo handling
    void loadLogo(const std::string& logoPath, bool = false);
    void drawLogo();
    
    // Output methods
    void saveAsPNG(const std::string& filename, const std::string& folder);
    
    // Legacy chart setup helper - consider using chart builder instead
    void setupChart(const ChartConfig& config, unsigned int* originalTopMargin = NULL,
                   unsigned int* originalRightMargin = NULL, unsigned int newRightMargin = 180,
                   int* titleY = NULL, int* legendY = NULL);
    
    // Simplified visualization methods
    void plotChart(const std::vector<Series>& seriesList,
                  const std::string& title = "Chart Visualization",
                  const std::string& xAxisLabel = "X Value",
                  const std::string& yAxisLabel = "Y Value");
         
    void plotChart(const std::vector<Series>& seriesList,
                        const DataMapper::AxisRange& xRange,
                        const DataMapper::AxisRange& yRange,
			const std::vector<CandleData>& candles = std::vector<CandleData>());

         
    // Arrow visualization methods
    void plotArrows(const std::vector<Arrow>& arrows, bool redrawBackground = false);
    void plotArrow(const Arrow& arrow, bool redrawBackground = false);
    void plotArrow(double startX, double startY, double endX, double endY, 
                  const RGBA& color = RGBA(), int lineWidth = 2, int arrowheadSize = 10,
                  bool redrawBackground = false);
    
    // Original method signatures needed for backward compatibility
    void plotHistogram(const std::vector<int>& bins, const RGBA& color = RGBA(), bool showXAxisLabels = true);
    void plotClusters(const std::vector<std::vector<double> >& data,
                     const std::vector<int>& labels,
                     const std::vector<std::vector<double> >& centroids);
    void plotCandlestickChart(const std::vector<CandleData>& candles,
                             const RGBA& bullishColor = RGBA(0x03, 0xC0, 0x3C, 0xFF),
                             const RGBA& bearishColor = RGBA(0xFF, 0x47, 0x45, 0xFF));
    
    // Enhanced methods with additional parameters
    void plotHistogram(const std::vector<int>& bins, 
                      const RGBA& color, 
                      bool showXAxisLabels,
                      const std::string& title,
                      const std::string& xAxisLabel,
                      const std::string& yAxisLabel);
                      
    // New method for histograms with custom labels
    void plotHistogramWithLabels(const std::vector<int>& bins,
                               const std::vector<std::string>& labels,
                               const RGBA& color = RGBA(),
                               const std::string& title = "Data Distribution",
                               const std::string& xAxisLabel = "Value",
                               const std::string& yAxisLabel = "Frequency");
                      
    void plotCandlestickChart(const std::vector<DataMapper::CandleData>& candles,
                             const RGBA& bullishColor,
                             const RGBA& bearishColor,
                             const std::string& title,
                             const std::string& xAxisLabel,
                             const std::string& yAxisLabel);
                              
    void plotClusters(const std::vector<std::vector<double> >& data,
                     const std::vector<int>& labels,
                     const std::vector<std::vector<double> >& centroids,
                     const std::string& title,
                     const std::string& xAxisLabel,
                     const std::string& yAxisLabel);
    
    // Control centroid alignment with cluster centers (public API)
    void setAlignCentroidsWithClusters(bool align);
    bool getAlignCentroidsWithClusters() const;
    
    // Direct access to GridRenderer for Y-axis ticks - consider using chart builder instead
    GridRenderer& getGridRenderer() { return *gridRenderer; }
    
private:
    friend class ChartBuilder;
    
    RGBA bullishColor;
    RGBA bearishColor;
    int lastLegendY;
    

    // Component pointers
    ColorManager* colorManager;
    SuperSamplingManager* ssaaManager;
    ChartLayout* chartLayout;
    ShapeRenderer* shapeRenderer;
    TextRenderer* textRenderer;
    GridRenderer* gridRenderer;
    DataMapper* dataMapper;
    ChartStyler* chartStyler;

    // font handling
    static FT_Library ft;
    static FT_Face face;
    static bool fontLoaded;
    
    // Output image
    Image image;
    
    // Logo handling
    static bool logoLoaded;
    static bool hasLogo;
    static Image logoImage;
    static int logoWidth;
    static int logoHeight;
    static int logoWidthScaled;
    static int logoHeightScaled;
    
    // Data axis ranges for origin axes
    DataMapper::AxisRange currentXAxisRange;
    DataMapper::AxisRange currentYAxisRange;
    
    // Saved margin state for restoration
    unsigned int savedMarginTop;
    unsigned int savedMarginRight;
    unsigned int savedMarginBottom;
    unsigned int savedMarginLeft;
    
    // Stored chart title information
    std::string chartTitle;
    unsigned int chartTitleFontSize;
    
    // Whether to force centroids to align with calculated cluster centers
    bool forceAlignCentroids;
    
    // Margin calculation helpers
    unsigned int calculateTopMargin(ChartType chartType, unsigned int width, unsigned int height);
    unsigned int calculateRightMargin(ChartType chartType, unsigned int width, unsigned int height);
    unsigned int calculateBottomMargin(ChartType chartType, unsigned int width, unsigned int height);
    unsigned int calculateLeftMargin(ChartType chartType, unsigned int width, unsigned int height);
    
    // Helper methods - general
    std::vector<DataMapper::Point> convertToDataPoints(const std::vector<Point>& points);
    void calculateDataRanges(const std::vector<DataMapper::Point>& dataPoints, 
                           DataMapper::AxisRange& xRange, 
                           DataMapper::AxisRange& yRange);
    void setupStandardAxisTicks(const DataMapper::AxisRange& xRange, 
                              const DataMapper::AxisRange& yRange, 
                              int yLabelOffset);
    bool isCoordinateValid(int x, int y);
    int clampToViewport(int value, int max);
    void drawLineSegment(int x1, int y1, int x2, int y2, const RGBA& color, int width, size_t segmentIndex);
    void prepareStandardChart(const std::string& title, 
                            const std::string& xAxisLabel, 
                            const std::string& yAxisLabel,
                            unsigned int titleFontSize,
                            unsigned int axisFontSize,
                            unsigned int rightMargin,
                            unsigned int* originalTopMargin,
                            unsigned int* originalRightMargin,
                            int* titleY,
                            int* legendY);
    
    // Helper methods - histogram specific
    void calculateHistogramBarDimensions(int totalBars, int& barWidth, int& barSpacing, int& startX);
    
    // Helper methods - candlestick specific
    std::vector<DataMapper::CandleData> convertToCandleData(const std::vector<CandleData>& candles);
    void calculateCandlestickRanges(const std::vector<DataMapper::CandleData>& candles,
                                  DataMapper::AxisRange& timeRange,
                                  DataMapper::AxisRange& priceRange);
    std::vector<DataMapper::CandleXAxis> createTimeLabels(double minTime, double maxTime,int numLabels);
    
    // Helper methods - cluster specific
    std::vector<RGBA> prepareClusterColors(int numClusters);
    std::vector<std::string> createClusterLegendLabels(int numClusters);
};

} // namespace shmea

#endif // PLOTTER_H 
