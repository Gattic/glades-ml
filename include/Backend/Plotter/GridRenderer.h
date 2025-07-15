#ifndef GRID_RENDERER_H
#define GRID_RENDERER_H

#include "../Database/image.h"
#include "ColorManager.h"
#include "SuperSamplingManager.h"
#include "ChartLayout.h"
#include "ShapeRenderer.h"
#include "DataMapper.h"
#include "Plotter.h"
#include <ft2build.h>
#include FT_FREETYPE_H

namespace shmea {

class GridRenderer {
public:
    enum XPositionMode 
    {
	X_START,
	X_CENTER,
	X_CUSTOM
    };
    // Constructor & Destructor
    GridRenderer(SuperSamplingManager& ssaaManager, ColorManager& colorManager, 
                ChartLayout& layout, ShapeRenderer& shapeRenderer, FT_Library&, FT_Face&);
    ~GridRenderer();
    
    // Draw the background with gradient and effects
    void drawBackground();
    
    // Draw grid lines
    void drawGrid();
    
    // Draw x and y axes
    void drawAxes();
    
    // Draw origin-centered axes for four quadrants
    void drawOriginAxes(double xMin, double xMax, double yMin, double yMax);
    
    // Draw an info box with gradient background for legends, stats, etc.
    void drawInfoBox(int x, int y, int boxWidth, int boxHeight, const std::string& text, unsigned int fontSize = 16);
    
    // Draw X-axis and Y-axis tick marks and labels
    void drawCandleXAxisTicks(const std::vector<DataMapper::CandleData>& candles, const std::vector<DataMapper::CandleXAxis>& labels, int startingOffset=1, int candleOffset=0);
    void drawXAxisTicks(const std::vector<DataMapper::CandleXAxis>& labels, const Series& series);
    void drawXAxisTicks(const std::vector<std::string>& labels, int numTicks, XPositionMode xMode = X_START, int startingOffset = 0);
    void drawXAxisTicks(double minValue, double maxValue, int numTicks, int precision = 1);
    void drawYAxisTicks(double minValue, double maxValue, int numTicks, bool isInteger = false, 
                      int precision = 1, int labelOffset = 25);
    
private:
    // References to other components
    SuperSamplingManager& ssaa;
    ColorManager& colors;
    ChartLayout& layout;
    ShapeRenderer& shapes;

    FT_Library ft;
    FT_Face face;
};

} // namespace shmea

#endif // GRID_RENDERER_H 
