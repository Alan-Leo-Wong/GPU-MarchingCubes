cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Failed to detect a default CUDA architecture. Use '-DCMAKE_CUDA_ARCHITECTURES=XX' to specify the default value.")
endif()

project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mc STATIC MarchingCubes.cu)
target_compile_features(mc PRIVATE cuda_std_14)
target_include_directories(mc PRIVATE include)

add_executable(test test.cpp)
target_compile_features(test PRIVATE cxx_std_14)

target_link_libraries(test PUBLIC mc)