#ifndef SHAPE_RENDERER_H
#define SHAPE_RENDERER_H

#include "../Database/image.h"
#include "ColorManager.h"
#include "SuperSamplingManager.h"
#include "ChartLayout.h"
#include <cmath>

namespace shmea {

class ShapeRenderer {
public:
    // Constructor & Destructor
    ShapeRenderer(SuperSamplingManager& ssaaManager, ColorManager& colorManager, ChartLayout& layout);
    ~ShapeRenderer();
    
    // Basic shape drawing functions
    void drawPoint(int x, int y, int size, const RGBA& color);
    void drawLine(int x1, int y1, int x2, int y2, const RGBA& lineColor, int lineWidth = 3);
    void drawRect(int x, int y, int width, int height, const RGBA& color, bool filled = true, int borderWidth = 1);
    void drawCircle(int x, int y, int radius, const RGBA& color, bool filled = true, int borderWidth = 1);
    
    // Advanced shape drawing
    void drawRoundedCorners(int left, int top, int right, int bottom, int radius, const RGBA& color);
    
    // Helper for blending a pixel with bounds checking
    void blendPixel(int x, int y, const RGBA& color, float alpha);
    
    // Specialized shapes for chart elements
    void drawCandlestick(int x, int y_open, int y_close, int y_high, int y_low, const RGBA& color, int customBodyWidth = -1);
    void drawHistogramBar(int x, int y, int barWidth, int barHeight, const RGBA& color);
    void drawHistogramBarHighlights(int x, int y, int barWidth, int barHeight);
    
    // Draw a rectangle with rounded corners
    void drawRoundedRect(int x, int y, int width, int height, int radius, const RGBA& color, bool filled = true);
    
    // Special solid rounded rectangle with perfect corner clipping - use this for UI elements
    void drawSolidRoundedRect(int x, int y, int width, int height, int radius, const RGBA& color);
    
    // Draw an arrow from (x1,y1) to (x2,y2) with arrowhead at end
    void drawArrow(int x1, int y1, int x2, int y2, const RGBA& color, int lineWidth = 2, int arrowheadSize = 10);
    
private:
    SuperSamplingManager& ssaa;
    ColorManager& colors;
    ChartLayout& layout;
};

} // namespace shmea

#endif // SHAPE_RENDERER_H 