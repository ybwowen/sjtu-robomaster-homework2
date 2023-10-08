cmake_minimum_required( VERSION 2.8 )
project(logo)

set(CMAKE_CXX_FLAGS "-std=c++14")

include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(logo main.cpp)
target_link_libraries(logo ${OpenCV_LIBS})