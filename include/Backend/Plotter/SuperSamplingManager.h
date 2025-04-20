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
    
    // Get supersampled dimensions
    unsigned int getWidth() const;
    unsigned int getHeight() const;
    
    // Get supersampling factor
    unsigned int getSamplingFactor() const;
    
    // Scaling helpers for coordinates and sizes
    int scaleX(int x) const;
    int scaleY(int y) const;
    int scaleSize(int size) const;
    
    // Access to the supersampled image
    Image& getImage();
    const Image& getImage() const;
    
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