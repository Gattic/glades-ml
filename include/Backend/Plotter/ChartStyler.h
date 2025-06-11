#ifndef CHART_STYLER_H
#define CHART_STYLER_H

#include "../Database/image.h"
#include "ColorManager.h"
#include "SuperSamplingManager.h"
#include "ChartLayout.h"
#include "ShapeRenderer.h"
#include "TextRenderer.h"
#include "GridRenderer.h"
#include "DataMapper.h"
#include <vector>
#include <string>
#include <utility> // for std::pair

namespace shmea {

class ChartStyler {
public:
    // Constructor & Destructor
    ChartStyler(ColorManager& colorManager, ChartLayout& layout,
               ShapeRenderer& shapeRenderer, TextRenderer& textRenderer,
               GridRenderer& gridRenderer, DataMapper& dataMapper);
    ~ChartStyler();
    
    // Create cluster legend and styling
    int createClusterLegend(const std::vector<RGBA>& clusterColors, int numClusters, int x, int y);
    
    // Helper for preparing legend colors and labels
    void prepareLegendColors(const std::vector<std::string>& labels, const std::vector<RGBA>& colors,
                           std::vector<std::string>& outLabels, std::vector<RGBA>& outColors);
    
    // Add a legend to the chart
    int addLegend(const std::vector<std::string>& labels, const std::vector<RGBA>& colors, 
                 int x, int y, unsigned int fontSize = 18, bool _new = false, bool redraw = false);
    
    // Histogram styling
    void drawHistogramStats(const std::vector<int>& bins, int maxBinValue, int legendY, unsigned int fontSize = 16);
    
    // Candlestick chart styling
    void drawCandlestickPriceInfo(const std::vector<DataMapper::CandleData>& candles,
                                const RGBA& bullishColor, const RGBA& bearishColor,
                                int legendY);
    
    // Calculate cluster bounds for visualization
    void calculateClusterBounds(
        const std::vector<std::vector<double> >& data,
        const std::vector<int>& labels,
        int cluster,
        std::vector<DataMapper::Point>& clusterCenters,
        std::vector<int>& clusterRadii,
        const DataMapper::AxisRange& xRange,
        const DataMapper::AxisRange& yRange);
        
    // Draw cluster visualization components
    void drawClusterCircles(
        const std::vector<DataMapper::Point>& clusterCenters,
        const std::vector<int>& clusterRadii,
        const std::vector<RGBA>& clusterColors);
        
    void drawCentroids(
        const std::vector<std::vector<double> >& centroids,
        const std::vector<RGBA>& clusterColors,
        const DataMapper::AxisRange& xRange,
        const DataMapper::AxisRange& yRange);
        
    void drawClusterLabels(
        const std::vector<DataMapper::Point>& clusterCenters,
        const std::vector<int>& clusterRadii,
        const std::vector<std::vector<std::pair<int, int> > >& clusterPoints,
        const std::vector<RGBA>& clusterColors);
    
    // Calculate the height needed for an info box
    int calculateInfoBoxHeight(const std::vector<std::string>& labels, unsigned int fontSize);
    
private:
    // References to other components
    ColorManager& colors;
    ChartLayout& layout;
    ShapeRenderer& shapes;
    TextRenderer& text;
    GridRenderer& grid;
    DataMapper& dataMapper;
    std::vector<std::string> legendLabels;
    std::vector<RGBA> legendColors;
};

} // namespace shmea

#endif // CHART_STYLER_H 
