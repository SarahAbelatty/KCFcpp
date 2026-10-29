#include "fhog.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main()
{
    const int width = 64;
    const int height = 64;
    const int cellSize = 4;

    IplImage* image = cvCreateImage(
        cvSize(width, height), IPL_DEPTH_8U, 3
    );

    if (image == nullptr) {
        std::printf("FAIL: Could not create test image.\n");
        return 1;
    }

    // Create a deterministic synthetic image.
    for (int y = 0; y < height; ++y) {
        unsigned char* row =
            reinterpret_cast<unsigned char*>(image->imageData
                                             + y * image->widthStep);

        for (int x = 0; x < width; ++x) {
            row[x * 3]     = static_cast<unsigned char>((x * 4) % 256);
            row[x * 3 + 1] = static_cast<unsigned char>((y * 4) % 256);
            row[x * 3 + 2] = static_cast<unsigned char>((x + y) * 2 % 256);
        }
    }

    CvLSVMFeatureMapCaskade* map = nullptr;

    int status = getFeatureMaps(image, cellSize, &map);

    if (status != 0 || map == nullptr) {
        std::printf("FAIL: getFeatureMaps returned %d.\n", status);
        cvReleaseImage(&image);
        return 1;
    }

    std::printf(
        "Before normalization: sizeX=%d, sizeY=%d, numFeatures=%d\n",
        map->sizeX, map->sizeY, map->numFeatures
    );

    bool valid = map->sizeX == width / cellSize
              && map->sizeY == height / cellSize
              && map->numFeatures == 27;

    const int initialCount = map->sizeX * map->sizeY * map->numFeatures;

    for (int i = 0; i < initialCount; ++i) {
        if (!std::isfinite(map->map[i])) {
            std::printf("FAIL: Non-finite HOG value at index %d.\n", i);
            valid = false;
            break;
        }
    }

    if (!valid) {
        std::printf("FAIL: Initial feature map validation.\n");
        freeFeatureMapObject(&map);
        cvReleaseImage(&image);
        return 1;
    }

    status = normalizeAndTruncate(map, VAL_OF_TRUNCATE);

    if (status != 0) {
        std::printf("FAIL: normalizeAndTruncate returned %d.\n", status);
        freeFeatureMapObject(&map);
        cvReleaseImage(&image);
        return 1;
    }

    std::printf(
        "After normalization: sizeX=%d, sizeY=%d, numFeatures=%d\n",
        map->sizeX, map->sizeY, map->numFeatures
    );

    valid = map->sizeX == width / cellSize - 2
         && map->sizeY == height / cellSize - 2
         && map->numFeatures == 108;

    const int outputCount = map->sizeX * map->sizeY * map->numFeatures;

    for (int i = 0; i < outputCount; ++i) {
        if (!std::isfinite(map->map[i]) ||
            map->map[i] < 0.0f ||
            map->map[i] > VAL_OF_TRUNCATE) {
            std::printf(
                "FAIL: Invalid normalized feature at index %d: %f\n",
                i, map->map[i]
            );
            valid = false;
            break;
        }
    }

    std::printf(
        "%s: HOG feature extraction and normalization validation.\n",
        valid ? "PASS" : "FAIL"
    );

    freeFeatureMapObject(&map);
    cvReleaseImage(&image);

    return valid ? 0 : 1;
}
