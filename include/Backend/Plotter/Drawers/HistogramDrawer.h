#ifndef HISTOGRAM_DRAWER_H
#define HISTOGRAM_DRAWER_H

#include "BaseDrawer.h"
#include <vector>

namespace shmea {

class HistogramDrawer : public BaseDrawer {
private:
    double min_price;
    double max_price;
    int graphSize;

public:
    HistogramDrawer(Image& image, unsigned int width, unsigned int height, 
                   int margin_top, int margin_right, int margin_bottom, int margin_left,
                   double min_price, double max_price, int graphSize);

    void drawBar(int x_start, int y_start, int bar_width, const RGBA& barColor);
    void addHistogram(const std::vector<int>& bins, const RGBA& barColor);
};

}  // namespace shmea
#endif
