# GPU Marching Cubes

This repository contains a GPU-accelerated implementation of the Marching Cubes algorithm, designed to extract a triangle mesh of an isosurface from a three-dimensional scalar field (volume data). The implementation utilizes CUDA (>=11.6) to perform calculations on the GPU, offering significant performance advantages over CPU-based solutions.

## Features

- **High Performance**: Utilizes the computational power of modern GPUs to accelerate the Marching Cubes algorithm.
- **Customizable SDF**: Supports custom Signed Distance Functions (SDF) for isosurface generation, allowing for diverse geometric shapes.
- **Configurable Resolution**: Easy configuration of the voxel grid resolution, origin, and width through the `Args` struct.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher.
- CUDA Toolkit 11.6 or higher.
- C++ compiler compatible with the CUDA version used (C++17 and cuda_std_17 or higher).
- CMake(>=3.18) for building the project.

## Configuration

Before compiling the project, you must configure the Marching Cubes parameters and the Signed Distance Function (SDF):

### 1. Modify Parameters in `main.cpp`

Set the grid parameters and isovalue in `main.cpp`:

```cpp
struct Args{
    uint3 resolution = make_uint3(200, 200, 200);
    double3 gridOrigin = make_double3(-2, -2, -2);
    double3 gridWidth = make_double3(4, 4, 4);
    double isoVal = 1.0; // Isovalue of level set
};
```

### 2. Set the Signed Distance Function (SDF)

We provide two interfaces for sdf computation.

#### 2.1 CPU Version:

Modify the `computeSDF` function in `main.cpp` to change the SDF computation:

```cpp
double computeSDF(const double3 &pos) {
    // Example: A sphere function
    return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
}
```

##### 2.2 GPU Version:

For computations directly on the GPU, modify the `computeSDF` function in `MarchingCubes.cu`:

```c++
__device__ double MCKernel::computeSDF(double3 pos) {
    // Example: A sphere function
    return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
}
```

## Compilation

```
mkdir build
cd build
cmake ..
cmake --build . -j your-core-num
```

This will compile the project into an executable named `main`. If there are any errors during the compilation, ensure that CMakeLists.txt is correctly set up to find CUDA and compile the .cu files.

## Usage

Run the executable with the output path for the generated mesh:

```
./main output-path.obj
```

This command will generate a triangle mesh based on the configured SDF and save it to the specified `.obj` file.

The example code generates a mesh for a sphere with a radius of 1.
