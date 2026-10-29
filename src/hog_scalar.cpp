#include "hog_scalar.hpp"

void applyHanningScalar(
    const float* features,
    const float* hann,
    float* output,
    int rows,
    int cols
) {
    if (features == nullptr || hann == nullptr || output == nullptr ||
        rows <= 0 || cols <= 0) {
        return;
    }

    const int totalElements = rows * cols;

    for (int i = 0; i < totalElements; ++i) {
        output[i] = features[i] * hann[i];
    }
}
