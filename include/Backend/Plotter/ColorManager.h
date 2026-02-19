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

    // Blend colors - inline for performance (called millions of times)
    inline RGBA blendColors(const RGBA& baseColor, const RGBA& overlayColor, float alpha) const {
        float oneMinusAlpha = 1.0f - alpha;
        return RGBA(
            static_cast<unsigned char>(baseColor.r * oneMinusAlpha + overlayColor.r * alpha),
            static_cast<unsigned char>(baseColor.g * oneMinusAlpha + overlayColor.g * alpha),
            static_cast<unsigned char>(baseColor.b * oneMinusAlpha + overlayColor.b * alpha),
            baseColor.a  // Keep the original alpha
        );
    }

    inline RGBA blendRGBA(const RGBA& base, const RGBA& over) const {
        // If the overlay is fully transparent, return the base unchanged
        if (over.a == 0) return base;

        // If the overlay is fully opaque, return it directly
        if (over.a == 255) return over;

        // Precompute reciprocal to avoid repeated division
        static const float inv255 = 1.0f / 255.0f;

        // Calculate alpha values for blending
        float alphaOver = over.a * inv255;
        float alphaBase = base.a * inv255;
        float alphaOut = alphaOver + alphaBase * (1.0f - alphaOver);

        // If the resulting alpha is zero, return transparent black
        if (alphaOut < 0.001f) return RGBA(0, 0, 0, 0);

        // Single reciprocal instead of three divisions
        float invAlphaOut = 1.0f / alphaOut;
        float baseContrib = alphaBase * (1.0f - alphaOver);

        // Blend the colors properly considering the alpha channels
        unsigned char r = static_cast<unsigned char>((over.r * alphaOver + base.r * baseContrib) * invAlphaOut);
        unsigned char g = static_cast<unsigned char>((over.g * alphaOver + base.g * baseContrib) * invAlphaOut);
        unsigned char b = static_cast<unsigned char>((over.b * alphaOver + base.b * baseContrib) * invAlphaOut);
        unsigned char a = static_cast<unsigned char>(alphaOut * 255.0f);

        return RGBA(r, g, b, a);
    }

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
