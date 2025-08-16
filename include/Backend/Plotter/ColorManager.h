#ifndef COLOR_MANAGER_H
#define COLOR_MANAGER_H

#include "../Database/image.h"
#include <map>
#include <string>
#include <vector>

namespace shmea {

class ColorManager {
public:
    // Constructor & Destructor
    ColorManager();
    ~ColorManager();
    
    // Initialize theme colors
    void initialize();
    
    // Initialize a special 10-cluster color scheme
    void initialize10ClusterScheme();
    
    // Get theme colors
    RGBA getThemeColor(int index) const;
    RGBA getElementColor(const std::string& element) const;
    
    // Check if an element color exists
    bool hasElementColor(const std::string& element) const;
    
    // Blend colors
    RGBA blendColors(const RGBA& baseColor, const RGBA& overlayColor, float alpha) const;
    RGBA blendRGBA(const RGBA& base, const RGBA& over) const;
    
    //Color Generation  
    RGBA HSVtoRGBA(float h, float s, float v);
    
    // Get color collections
    const std::vector<RGBA>& getThemeColors() const;
    const std::map<std::string, RGBA>& getElementColors() const;
    
    // Set theme colors
    void setThemeColors(const std::vector<RGBA>& colors);

private:
    std::vector<RGBA> themeColors;
    std::map<std::string, RGBA> elementColors;
};

} // namespace shmea

#endif // COLOR_MANAGER_H 
