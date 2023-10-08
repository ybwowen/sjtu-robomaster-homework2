cmake_minimum_required( VERSION 2.8 )
project(photograph)

set(CMAKE_CXX_FLAGS "-std=c++14")

include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(photograph main.cpp)
target_link_libraries(photograph ${OpenCV_LIBS})