#ifndef ARROW_DRAWER_H
#define ARROW_DRAWER_H

#include "BaseDrawer.h"
#include <vector>

namespace shmea {

class ArrowDrawer : public BaseDrawer {
public:
    ArrowDrawer(Image& image, unsigned int width, unsigned int height, 
                int margin_top, int margin_right, int margin_bottom, int margin_left);
    
    void drawArrow(int x1, int y1, int x2, int y2, const RGBA& arrowColor, int arrowSize = 10);
    void addArrow(const std::vector<std::vector<double> >& sorted_eig_vecs, 
                  const std::vector<double>& variance_explained, const RGBA& arrowColor);
};

}  // namespace shmea
#endif
