#ifndef DATA_DRAWER_H
#define DATA_DRAWER_H

#include "BaseDrawer.h"
#include <vector>
#include <map>
#include <string>
#include <limits>

namespace shmea {

class DataDrawer : public BaseDrawer {
private:
    // Line tracking
    int lines;
    std::vector<bool> first_line_point;
    std::vector<int> last_price_pos;
    int last_line_drawn;
    std::vector<RGBA> line_colors;
    std::vector<std::string> line_color_names;
    
    // Indicator colors
    std::map<std::string, RGBA> indicatorColors;
    std::map<std::string, RGBA> indicatorTextColor;
    std::map<std::string, int> indicatorPoint;
    
    // Dimensions
    double min_price;
    double max_price;
    int graphSize;
    int candle_width;
    
    // Initialize colors
    void initialize_colors();

public:
    DataDrawer(Image& image, unsigned int width, unsigned int height, 
               int margin_top, int margin_right, int margin_bottom, int margin_left,
               double min_price, double max_price, int graphSize, int lines);
               
    void addDataPoint(double newPrice, int portIndex, bool draw, RGBA* lineColor = NULL, int lineWidth = 6);
    void addDataPointWithIndicator(double newPrice, int portIndex, const std::string& indicator, const std::string& value);
    void addDataPointsPCA(const std::vector<std::vector<double> >& data, const RGBA& pointColor);
    void addDataPointsKMeans(const std::string& graphName, 
                            const std::vector<std::vector<double> >& data, 
                            const std::vector<int>& labels, 
                            const std::vector<std::vector<float> >& centroids);
                            
    void drawCentroidCircle(int x, int y, int radius, const RGBA& color);
    void drawClusterCircle(int x, int y, int radius, const RGBA& color);
    
    // Getter for indicator colors
    const RGBA* getIndicatorColor(const std::string& indicator) const;
    const RGBA* getIndicatorTextColor(const std::string& indicator) const;
};

} // namespace shmea
#endif
