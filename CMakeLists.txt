cmake_minimum_required(VERSION 3.18)

project(MarchingCubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MC STATIC MarchingCubes.h MarchingCubes.cu)
target_compile_features(MC PRIVATE cuda_std_17)

add_executable(main main.cpp)
target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main PUBLIC MC)