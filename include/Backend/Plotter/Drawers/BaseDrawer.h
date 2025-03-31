#ifndef BASE_DRAWER_H
#define BASE_DRAWER_H

#include "../../Database/image.h"
#include <vector>

namespace shmea {

class BaseDrawer {
protected:
    Image& image;
    unsigned int width;
    unsigned int height;
    int margin_top;
    int margin_right;
    int margin_bottom;
    int margin_left;
    
    // Helper methods
    void drawPoint(int x, int y, int thickness, const RGBA& pointColor);
    void drawLine(int x1, int y1, int x2, int y2, const RGBA& lineColor, int lineWidth = 6);
    void drawCircle(int x, int y, int radius, const RGBA& color);
    void drawCirclePoints(int x, int y, int x0, int y0, const RGBA& color);
    
    // Utility functions
    int clamp(int value, int min, int max) const;
    
public:
    BaseDrawer(Image& image, unsigned int width, unsigned int height, 
               int margin_top, int margin_right, int margin_bottom, int margin_left);
    virtual ~BaseDrawer() {}; // Changed from "= default;" to empty implementation for C++03
};

}  // namespace shmea
#endif
