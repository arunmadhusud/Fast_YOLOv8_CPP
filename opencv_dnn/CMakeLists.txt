cmake_minimum_required(VERSION 3.12)
project(inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR /home/arun/opencv/build)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(inference main.cpp inference.cpp)

# Link libraries
target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${OpenCV_LIBRARIES})