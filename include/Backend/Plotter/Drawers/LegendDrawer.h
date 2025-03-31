#ifndef LEGEND_DRAWER_H
#define LEGEND_DRAWER_H

#include "BaseDrawer.h"
#include <vector>
#include <string>
#include <utility>

namespace shmea {

class LegendDrawer : public BaseDrawer {
private:
    struct LegendEntry {
        std::string label;
        RGBA color;
        
        LegendEntry(const std::string& l, const RGBA& c) : label(l), color(c) {}
    };
    
    std::vector<LegendEntry> entries;
    unsigned int entry_height;
    unsigned int entry_spacing;
    unsigned int box_size;
    unsigned int font_size;
    unsigned int padding;
    RGBA background_color;
    RGBA text_color;
    RGBA border_color;
    unsigned int max_height; // Maximum height constraint for the legend
    
    void drawLegendBox(int x, int y, int width, int height);
    void drawEntryBox(int x, int y, const RGBA& color);
    void drawEntryLabel(int x, int y, const std::string& label);

public:
    LegendDrawer(Image& image, unsigned int width, unsigned int height, 
                int margin_top, int margin_right, int margin_bottom, int margin_left);
    
    void addEntry(const std::string& label, const RGBA& color);
    void draw(unsigned int x, unsigned int y);
    
    // Setters for customizing the legend appearance
    void setEntryHeight(unsigned int height);
    void setEntrySpacing(unsigned int spacing);
    void setBoxSize(unsigned int size);
    void setFontSize(unsigned int size);
    void setPadding(unsigned int pad);
    void setBackgroundColor(const RGBA& color);
    void setTextColor(const RGBA& color);
    void setBorderColor(const RGBA& color);
    void setMaxHeight(unsigned int height);
};

} // namespace shmea
#endif
