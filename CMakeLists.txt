cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
    ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src)

FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

add_executable(KCF ${sourcefiles})
target_link_libraries(KCF ${OpenCV_LIBS})

# Standalone HOG feature extraction test
add_executable(test_hog_features
    test_hog_features.cpp
    src/fhog.cpp
)
target_link_libraries(test_hog_features ${OpenCV_LIBS})

add_executable(test_hanning
    test_hanning.cpp
    src/hog_scalar.cpp
)
target_link_libraries(test_hanning ${OpenCV_LIBS})