#  minimum required cmake version: 3.17
cmake_minimum_required(VERSION 3.17)

project(RealsenseStabilization)

find_package(OpenCV REQUIRED)

set(DEPENDENCIES realsense2 ${OpenCV_LIBS})

add_executable(${PROJECT_NAME} RSstab.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})