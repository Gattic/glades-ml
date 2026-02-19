#ifndef SUPER_SAMPLING_MANAGER_H
#define SUPER_SAMPLING_MANAGER_H

#include "../Database/image.h"

namespace shmea {

class SuperSamplingManager {
public:
    // Constructor & Destructor
    SuperSamplingManager(unsigned int width, unsigned int height, unsigned int ssaaFactor);
    ~SuperSamplingManager();

    // Initialize supersampling buffer
    void initializeBuffer();

    // Downsample from supersampled image to output image
    void downsampleToOutput(Image& outputImage);

    // Set new supersampling factor
    void setSuperSamplingFactor(unsigned int factor);

    // Get supersampled dimensions - inline for performance
    unsigned int getWidth() const { return ssaaWidth; }
    unsigned int getHeight() const { return ssaaHeight; }

    // Get supersampling factor - inline for performance
    unsigned int getSamplingFactor() const { return ssaaFactor; }

    // Scaling helpers for coordinates and sizes - inline for performance
    int scaleX(int x) const { return x * ssaaFactor; }
    int scaleY(int y) const { return y * ssaaFactor; }
    int scaleSize(int size) const { return size * ssaaFactor; }

    // Access to the supersampled image - inline for performance
    Image& getImage() { return ssaaImage; }
    const Image& getImage() const { return ssaaImage; }

private:
    unsigned int originalWidth;
    unsigned int originalHeight;
    unsigned int ssaaFactor;
    unsigned int ssaaWidth;
    unsigned int ssaaHeight;
    Image ssaaImage;
};

} // namespace shmea

#endif // SUPER_SAMPLING_MANAGER_H