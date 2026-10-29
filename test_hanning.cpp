
#include <cmath>
#include <iostream>
#include "src/hog_scalar.hpp"
#include <opencv2/core.hpp>

int main() {
    const int rows = 2;
    const int cols = 3;
    const float tolerance = 1e-6f;

    const float features[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const float hann[] = {
        0.0f, 0.5f, 1.0f,
        1.0f, 0.5f, 0.0f
    };

    float output[rows * cols] = {};

    applyHanningScalar(features, hann, output, rows, cols);

    // Build OpenCV matrices from the same input data.
    cv::Mat featureMat(rows, cols, CV_32F);
    cv::Mat hannMat(rows, cols, CV_32F);

    for (int i = 0; i < rows * cols; ++i) {
        featureMat.ptr<float>()[i] = features[i];
        hannMat.ptr<float>()[i] = hann[i];
    }

    cv::Mat reference = featureMat.mul(hannMat);

    for (int i = 0; i < rows * cols; ++i) {
        const float expected = reference.ptr<float>()[i];

        if (!std::isfinite(output[i]) ||
            std::fabs(output[i] - expected) > tolerance) {
            std::cerr << "FAIL at index " << i
                      << ": scalar=" << output[i]
                      << ", OpenCV=" << expected << '\n';
            return 1;
        }
    }

    std::cout
        << "PASS: Scalar Hanning multiplication matches OpenCV.\n";

    return 0;
}
