#ifndef HOG_SCALAR_HPP
#define HOG_SCALAR_HPP

// Element-wise multiplication of HOG features by the Hanning window.
// Both arrays use row-major contiguous storage.
void applyHanningScalar(
    const float* features,
    const float* hann,
    float* output,
    int rows,
    int cols
);

#endif
