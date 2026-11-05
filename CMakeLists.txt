cmake_minimum_required(VERSION 3.0)
project(KCF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} src)

# Explicitly list the files so CMake knows exactly what to build
add_executable(KCF 
    src/runtracker.cpp 
    src/kcftracker.cpp 
    src/fhog.cpp
    src/resize_bilinear_scalar.cpp
)

target_link_libraries(KCF ${OpenCV_LIBS} pthread m)