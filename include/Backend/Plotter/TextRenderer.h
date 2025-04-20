#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include "../Database/image.h"
#include "ColorManager.h"
#include "SuperSamplingManager.h"
#include "ChartLayout.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <string>
#include <vector>

namespace shmea {

class TextRenderer {
public:
    // Constructor & Destructor
    TextRenderer(SuperSamplingManager& ssaaManager, ColorManager& colorManager, ChartLayout& layout);
    ~TextRenderer();
    
    // Initialize font
    void initialize(FT_Library&, FT_Face&);
    
    // Draw text at specified position (origin point is left-center of text)
    void drawText(int x, int y, const std::string& text, const RGBA& color, 
                 unsigned int fontSize = 18, bool centerAligned = false);
    
    // Draw text rotated for y-axis labels
    void drawVerticalText(const std::string& text, int x, int y, int fontSize, const RGBA& color);
    
    // Add title to the visualization
    void addTitle(const std::string& text, unsigned int fontSize = 36);
    
    // Add axis labels
    void drawAxisLabels(const std::string& xLabel, const std::string& yLabel, 
                       unsigned int fontSize, bool centerX = true);
    
    // Set Y axis label (stored for later use)
    void setYAxisLabel(const std::string& label);
    
    // Get current Y axis label
    const std::string& getYAxisLabel() const;
    
    // Set axis label font size
    void setAxisLabelFontSize(unsigned int fontSize);
    
    // Get axis label font size
    unsigned int getAxisLabelFontSize() const;
    
private:
    // FreeType variables
    FT_Library ft;
    FT_Face face;
    
    // References to other components
    SuperSamplingManager& ssaa;
    ColorManager& colors;
    ChartLayout& layout;
    
    // State
    std::string yAxisLabel;
    bool fontInitialized;
    
    // Font sizes
    unsigned int axisLabelFontSize;
};

} // namespace shmea

#endif // TEXT_RENDERER_H 
