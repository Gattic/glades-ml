#ifndef HISTOGRAM_DRAWER_H
#define HISTOGRAM_DRAWER_H

#include "BaseDrawer.h"
#include "LabelsDrawer.h"
#include <vector>
#include <string>

namespace shmea {

class HistogramDrawer : public BaseDrawer {
private:
    double min_price;
    double max_price;
    int graphSize;
    LabelsDrawer* labelsDrawer;
    
    // Styling properties
    int bar_spacing;
    int max_bar_height;
    bool show_grid_lines;
    bool show_values;
    bool show_bar_labels;
    int grid_line_count;
    RGBA grid_line_color;
    unsigned int bar_label_font_size;
    RGBA bar_label_color;
    
    // Helper methods
    void drawBar(int x_start, int y_start, int bar_width, int bar_height, const RGBA& barColor);
    void drawHorizontalGridLines(int max_count);
    void drawValueLabel(int x_start, int bar_width, int y_start, int value);
    void drawBarLabel(int x_start, int bar_width, const std::string& label);
    std::string formatNumber(int number) const;

public:
    HistogramDrawer(Image& image, unsigned int width, unsigned int height, 
                   int margin_top, int margin_right, int margin_bottom, int margin_left,
                   double min_price, double max_price, int graphSize);
    
    // Set the LabelsDrawer reference
    void setLabelsDrawer(LabelsDrawer* drawer);
    
    // Configure histogram appearance
    void setBarSpacing(int spacing);
    void setShowGridLines(bool show);
    void setShowValues(bool show);
    void setShowBarLabels(bool show);
    void setGridLineCount(int count);
    void setGridLineColor(const RGBA& color);
    void setBarLabelFontSize(unsigned int size);
    void setBarLabelColor(const RGBA& color);
    
    // Main drawing methods
    void addHistogram(const std::vector<int>& bins, const RGBA& barColor);
    void addHistogramWithLabels(const std::vector<int>& bins, const RGBA& barColor, 
                               const std::vector<std::string>& labels);
};

}  // namespace shmea
#endif
