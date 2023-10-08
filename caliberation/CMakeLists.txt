cmake_minimum_required( VERSION 2.8 )
project(caliberation)

set(CMAKE_CXX_FLAGS "-std=c++14")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(caliberation main.cpp)
target_link_libraries(caliberation ${OpenCV_LIBS})