cmake_minimum_required(VERSION 3.18)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# message(STATUS "Failed to detect a default CUDA architecture. Use '-DCMAKE_CUDA_ARCHITECTURES=XX' to specify the default value.")
# set(CMAKE_CUDA_ARCHITECTURES 86)
# endif()
project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MC STATIC MarchingCubes.h MarchingCubes.cu)
target_compile_features(MC PRIVATE cuda_std_14)

# target_include_directories(MC PRIVATE include)
add_executable(TEST test.cpp)
target_compile_features(TEST PRIVATE cxx_std_14)

target_link_libraries(TEST PUBLIC MC)