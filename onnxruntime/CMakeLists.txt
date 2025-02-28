cmake_minimum_required(VERSION 3.12)
project(inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the location of ONNX Runtime headers and libraries
set(ONNX_RUNTIME_INCLUDE_DIR "/home/arun/Adv_CV/onnx_rt_new/onnxruntime/include/onnxruntime/core/session")
set(ONNX_RUNTIME_LIB_DIR "/home/arun/Adv_CV/onnx_rt_new/onnxruntime/build/Linux/RelWithDebInfo")

# Add the include directory for ONNX Runtime
# include_directories(${ONNX_RUNTIME_INCLUDE_DIR})

# Find the ONNX Runtime library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    HINTS ${ONNX_RUNTIME_LIB_DIR}
    PATH_SUFFIXES lib
    REQUIRED
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found. Please check the ONNX_RUNTIME_LIB_DIR path.")
endif()

set(OpenCV_DIR /home/arun/opencv/build)

find_package(OpenCV 4 REQUIRED)

# Add the executable
add_executable(inference main.cpp inference.cpp)
# add_executable(inference check.cpp)

# Link libraries
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

# Link to ONNX Runtime
target_link_libraries(inference PRIVATE ${ONNXRUNTIME_LIB} ${OpenCV_LIBRARIES})

# Print some debug information
message(STATUS "ONNX Runtime Include Dir: ${ONNX_RUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime Library Dir: ${ONNX_RUNTIME_LIB_DIR}")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIB}")
