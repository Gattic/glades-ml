#ifndef CHART_LAYOUT_H
#define CHART_LAYOUT_H

#include <string>
#include <vector>
#include "../Database/image.h"

namespace shmea {

class ChartLayout {
public:
    // Chart configuration structure
    struct ChartConfig {
        std::string title;
        unsigned int titleFontSize;
        std::string xAxisLabel;
        std::string yAxisLabel;
        unsigned int axisFontSize;
        
        ChartConfig() 
            : title(""), titleFontSize(36), xAxisLabel(""), yAxisLabel(""), axisFontSize(24) {}
        
        ChartConfig(const std::string& title_, unsigned int titleSize, 
                   const std::string& xLabel, const std::string& yLabel, unsigned int axisSize = 24)
            : title(title_), titleFontSize(titleSize), xAxisLabel(xLabel), yAxisLabel(yLabel), axisFontSize(axisSize) {}
    };

    // Constructor & Destructor
    ChartLayout(unsigned int width, unsigned int height, 
               unsigned int margin_top, unsigned int margin_right, 
               unsigned int margin_bottom, unsigned int margin_left,
               unsigned int ssaa_factor = 1);
    ~ChartLayout();
    
    // Get dimensions and margins
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getMarginTop() const;
    unsigned int getMarginRight() const;
    unsigned int getMarginBottom() const;
    unsigned int getMarginLeft() const;
    
    // Set margins
    void setMarginTop(unsigned int margin);
    void setMarginRight(unsigned int margin);
    void setMarginBottom(unsigned int margin);
    void setMarginLeft(unsigned int margin);
    
    // Calculate plotting area
    unsigned int getPlotWidth() const;
    unsigned int getPlotHeight() const;
    
    // Estimate text dimensions
    int estimateTextWidth(const std::string& text, unsigned int fontSize) const;
    int calculateInfoBoxHeight(const std::vector<std::string>& labels, unsigned int fontSize) const;
    
    // Corner radius for styling
    void setCornerRadius(int radius);
    int getCornerRadius() const;
    
    // Grid and axes visibility
    void setShowGrid(bool show);
    bool isGridVisible() const;
    void setShowAxes(bool show);
    bool areAxesVisible() const;
    
    // Origin axes (four quadrants) visibility
    void setShowOriginAxes(bool show);
    bool areOriginAxesVisible() const;
    
    // Supersampling factor
    unsigned int getSsaaFactor() const;
    void setSsaaFactor(unsigned int factor);
    
    // Access to supersampled image (needed for certain rendering operations)
    Image& getSsaaImage();
    const Image& getSsaaImage() const;
    
    // Logo dimensions
    int getLogoWidth() const;
    int getLogoHeight() const;
    void setLogoWidth(int width);
    void setLogoHeight(int height);
    
    // Helper method for maintaining integer bounds
    static int clamp(int value, int min, int max);
    
private:
    unsigned int width;
    unsigned int height;
    unsigned int marginTop;
    unsigned int marginRight;
    unsigned int marginBottom;
    unsigned int marginLeft;
    int cornerRadius;
    bool showGrid;
    bool showAxes;
    bool showOriginAxes;  // Whether to show origin-centered axes (four quadrants)
    unsigned int ssaaFactor;
    Image ssaaImage; // Reference to the supersampled image
    int logoWidth;  // Width of the logo image
    int logoHeight; // Height of the logo image
};

} // namespace shmea

#endif // CHART_LAYOUT_H 