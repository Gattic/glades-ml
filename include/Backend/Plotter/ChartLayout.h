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
    
    // Get dimensions and margins — inline for performance
    unsigned int getWidth() const { return width; }
    unsigned int getHeight() const { return height; }
    unsigned int getMarginTop() const { return marginTop; }
    unsigned int getMarginRight() const { return marginRight; }
    unsigned int getMarginBottom() const { return marginBottom; }
    unsigned int getMarginLeft() const { return marginLeft; }

    // Set margins
    void setMarginTop(unsigned int margin) { marginTop = margin; }
    void setMarginRight(unsigned int margin) { marginRight = margin; }
    void setMarginBottom(unsigned int margin) { marginBottom = margin; }
    void setMarginLeft(unsigned int margin) { marginLeft = margin; }

    // Calculate plotting area — inline for performance
    unsigned int getPlotWidth() const { return width - marginLeft - marginRight; }
    unsigned int getPlotHeight() const { return height - marginTop - marginBottom; }

    // Estimate text dimensions
    int estimateTextWidth(const std::string& text, unsigned int fontSize) const;
    int calculateInfoBoxHeight(const std::vector<std::string>& labels, unsigned int fontSize) const;

    // Corner radius for styling
    void setCornerRadius(int radius) { cornerRadius = radius; }
    int getCornerRadius() const { return cornerRadius; }

    // Grid and axes visibility — inline for performance
    void setShowGrid(bool show) { showGrid = show; }
    bool isGridVisible() const { return showGrid; }
    void setShowAxes(bool show) { showAxes = show; }
    bool areAxesVisible() const { return showAxes; }

    // Origin axes (four quadrants) visibility
    void setShowOriginAxes(bool show) { showOriginAxes = show; }
    bool areOriginAxesVisible() const { return showOriginAxes; }

    //Decision if we want to display dates
    void setDateLabel(bool date) { dateLabel = date; }
    bool isDateLabelShown() const { return dateLabel; }

    //Decision if Legend is displayed
    void setLegendLabels(bool legend) { legendLabel = legend; }
    bool isLegendVisible() const { return legendLabel; }

    // Supersampling factor
    unsigned int getSsaaFactor() const { return ssaaFactor; }
    void setSsaaFactor(unsigned int factor);

    // Access to supersampled image (needed for certain rendering operations)
    Image& getSsaaImage() { return ssaaImage; }
    const Image& getSsaaImage() const { return ssaaImage; }

    // Logo dimensions
    int getLogoWidth() const { return logoWidth; }
    int getLogoHeight() const { return logoHeight; }
    void setLogoWidth(int w) { logoWidth = w; }
    void setLogoHeight(int h) { logoHeight = h; }

    // Helper method for maintaining integer bounds — inline for performance
    static int clamp(int value, int min, int max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
    
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
    bool dateLabel;
    bool legendLabel;
    unsigned int ssaaFactor;
    Image ssaaImage; // Reference to the supersampled image
    int logoWidth;  // Width of the logo image
    int logoHeight; // Height of the logo image
};

} // namespace shmea

#endif // CHART_LAYOUT_H 
