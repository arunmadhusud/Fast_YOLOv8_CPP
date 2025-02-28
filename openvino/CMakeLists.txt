cmake_minimum_required(VERSION 3.12)
project(yolov8_openvino_example)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV package
set(OpenCV_DIR /home/arun/opencv/build)

find_package(OpenCV 4 REQUIRED)

# Specify OpenVINO include directory manually
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /opt/intel/openvino_2024/runtime/include  # Update to the correct OpenVINO 2024 include path
)

# Add executable for your project
add_executable(inference  main.cc  inference.cc)

# Link the OpenCV and OpenVINO libraries
target_link_libraries(inference
    ${OpenCV_LIBS}
    /opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so  # Update to the correct OpenVINO 2024 library path
)
