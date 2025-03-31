#ifndef LABELS_DRAWER_H
#define LABELS_DRAWER_H

#include "BaseDrawer.h"
#include <string>
#include <vector>
#include <map>
#include <ft2build.h>
#include FT_FREETYPE_H

namespace shmea {

class LabelsDrawer : public BaseDrawer {
private:
    // Font handling
    FT_Library ft;
    FT_Face face;
    
    // Header layout settings
    unsigned int headerPenXStarting;
    unsigned int headerPenYStarting;
    unsigned int headerXSpacing;
    unsigned int headerYSpacing;
    std::vector<std::vector<unsigned int> > headerSpacings;
    
    // Initialization
    void initialize_font(const std::string& fontPath = "fonts/font.ttf");
    bool freetype_available;

public:
    LabelsDrawer(Image& image, unsigned int width, unsigned int height, 
                int margin_top, int margin_right, int margin_bottom, int margin_left);
    ~LabelsDrawer();
    
    // Text rendering methods
    void drawLabel(unsigned int penX, unsigned int penY, const std::string& text, 
                  unsigned int fontSize, unsigned int xOffset = 0, unsigned int yOffset = 0,
                  bool hasBox = false, RGBA labelColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF), 
                  RGBA textColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
    
    void drawHeader(const std::string& text, unsigned int fontSize, 
                   unsigned int headerPos = 0, unsigned int rePositionY = 0,
                   RGBA headerTextColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
                   
    // New text rendering methods for chart labels
    void drawCenteredText(unsigned int x, unsigned int y, const std::string& text,
                         unsigned int fontSize, RGBA textColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
                         
    void drawRotatedText(unsigned int x, unsigned int y, const std::string& text,
                        unsigned int fontSize, RGBA textColor = RGBA(0xFF, 0xFF, 0xFF, 0xFF));
};

}  // namespace shmea
#endif
