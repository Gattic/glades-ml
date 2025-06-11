#ifndef DATA_MAPPER_H
#define DATA_MAPPER_H

#include "ChartLayout.h"
#include <vector>
#include <limits>

namespace shmea {

class DataMapper {
public:
    // Data structures
    struct Point {
        double x;
        double y;
        Point() : x(0.0), y(0.0) {}
        Point(double x_, double y_) : x(x_), y(y_) {}
    };

    struct CandleData {
        double timestamp;  // Unix timestamp for the candle
        double open;
        double close;
        double high;
        double low;
        
        CandleData() : timestamp(0.0), open(0.0), close(0.0), high(0.0), low(0.0) {}
        CandleData(double t, double o, double c, double h, double l) 
            : timestamp(t), open(o), close(c), high(h), low(l) {}
    };

    struct CandleXAxis
    {
	double timestamp;
        std::string timeLabel;
        CandleXAxis() : timestamp(0.0), timeLabel("0") {}
        CandleXAxis(double t, std::string tl)
	    : timestamp(t), timeLabel(tl) {}


    };

    struct AxisRange {
        double min;
        double max;
        double padding;
        
        AxisRange() : min(0.0), max(1.0), padding(0.05) {}
        AxisRange(double min_, double max_, double padding_ = 0.05) 
            : min(min_), max(max_), padding(padding_) {}
    };
    
    // Constructor & Destructor
    DataMapper(ChartLayout& layout);
    ~DataMapper();
    
    // Calculate axis ranges for different data types
    AxisRange calculateXRange(const std::vector<Point>& points);
    AxisRange calculateYRange(const std::vector<Point>& points);
    AxisRange calculateXRange(const std::vector<std::vector<double> >& data);
    AxisRange calculateYRange(const std::vector<std::vector<double> >& data);
    
    // Map data coordinates to screen coordinates
    Point mapDataToScreen(double x, double y, const AxisRange& xRange, const AxisRange& yRange);
    
    // Get/set current axis ranges
    const AxisRange& getCurrentXRange() const { return currentXRange; }
    const AxisRange& getCurrentYRange() const { return currentYRange; }
    void setCurrentXRange(const AxisRange& range) { currentXRange = range; }
    void setCurrentYRange(const AxisRange& range) { currentYRange = range; }
    
private:
    // Functors for extracting values from different data types
    struct PointXValueFunctor {
        bool isEmptyData(const std::vector<Point>& data) const;
        size_t getSize(const std::vector<Point>& data) const;
        bool isValidIndex(const std::vector<Point>& data, size_t i) const;
        double getValue(const std::vector<Point>& data, size_t i) const;
    };
    
    struct PointYValueFunctor {
        bool isEmptyData(const std::vector<Point>& data) const;
        size_t getSize(const std::vector<Point>& data) const;
        bool isValidIndex(const std::vector<Point>& data, size_t i) const;
        double getValue(const std::vector<Point>& data, size_t i) const;
    };
    
    struct MatrixXValueFunctor {
        bool isEmptyData(const std::vector<std::vector<double> >& data) const;
        size_t getSize(const std::vector<std::vector<double> >& data) const;
        bool isValidIndex(const std::vector<std::vector<double> >& data, size_t i) const;
        double getValue(const std::vector<std::vector<double> >& data, size_t i) const;
    };
    
    struct MatrixYValueFunctor {
        bool isEmptyData(const std::vector<std::vector<double> >& data) const;
        size_t getSize(const std::vector<std::vector<double> >& data) const;
        bool isValidIndex(const std::vector<std::vector<double> >& data, size_t i) const;
        double getValue(const std::vector<std::vector<double> >& data, size_t i) const;
    };
    
    // Generic range calculation template
    template<typename DataType, typename ValueFunction>
    AxisRange calculateRange(const DataType& data, ValueFunction valueFunc);
    
    // Reference to layout
    ChartLayout& layout;
    
    // Current data ranges - used for auto alignment of origin axes
    AxisRange currentXRange;
    AxisRange currentYRange;
};

} // namespace shmea

#endif // DATA_MAPPER_H 
