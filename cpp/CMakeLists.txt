cmake_minimum_required(VERSION 3.16)
project(industrial_pcl_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation features search kdtree)

add_executable(pcl_pipeline pcl_pipeline.cpp)
target_include_directories(pcl_pipeline PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(pcl_pipeline PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(pcl_pipeline PRIVATE ${PCL_LIBRARIES})

