#ifndef CANDLESTICK_DRAWER_H
#define CANDLESTICK_DRAWER_H

#include "BaseDrawer.h"

namespace shmea {

class CandlestickDrawer : public BaseDrawer {
private:
    int candle_width;
    RGBA color_bullish;
    RGBA color_bearish;
    
public:
    CandlestickDrawer(Image& image, unsigned int width, unsigned int height, 
                      int margin_top, int margin_right, int margin_bottom, int margin_left,
                      int candle_width, const RGBA& bullish_color, const RGBA& bearish_color);
    
    void drawCandleStick(int x, int y_open, int y_close, int y_high, int y_low, const RGBA& color);
};

}  // namespace shmea
#endif
