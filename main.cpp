/*
 * @Author: Lei Wang
 * @Date: 2023-03-18 11:14:25
 * @LastEditors: Lei Wang
 * @LastEditTime: 2023-04-18 23:03:50
 * @FilePath: \GPUMarchingCubes\test.cpp
 */
#include "MarchingCubes.h"
#include "utils/Math.h"
#include <iostream>
#include <stdlib.h>
#include <execution>
#include <algorithm>

/// Define your SDF calculation here if you intend to compute the SDF on the CPU.
double computeSDF(const double3 &pos) {
    // here is a sphere function example
    return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
}

uint3 getVoxelShift(uint index,
                    const uint3 &res) {
    uint x = index % res.x;
    uint y = index % (res.x * res.y) / res.x;
    uint z = index / (res.x * res.y);
    return make_uint3(x, y, z);
}

struct Args{
    uint3 resolution = make_uint3(200, 200, 200);
    double3 gridOrigin = make_double3(-2, -2, -2); // the origin coordinate of voxel grid
    double3 gridWidth = make_double3(4, 4, 4);     // width of grid

    double isoVal = 1.0;
};

int main(int argc, char **argv) {
    Args args;

    double3 voxelSize = make_double3(args.gridWidth.x / args.resolution.x, args.gridWidth.y / args.resolution.y,
                                     args.gridWidth.z / args.resolution.z); // size of voxel
    const std::string outFile = argc > 1 ? argv[1] : "sphere.obj";

    /// 1. one way is calculating sdf on the GPU
    ///    you need to define your SDF on line 46 in "MarchingCubes.cu"
    {
        std::cout << "Test computing SDF on the gpu...\n";
        MC::marching_cubes(args.resolution, args.gridOrigin, voxelSize, args.isoVal, true, outFile);
        std::cout << "=================\n";
    }

    /// 2. another way is calculating sdf on the CPU, then transfer to the GPU
    /*{
        std::cout << "Test computing SDF on the cpu...\n";
        size_t nVoxels = resolution.x * resolution.y * resolution.z;
        std::vector<double> sdf(nVoxels * 8, 0);
        std::vector<size_t> indices(nVoxels);
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., nVoxels - 1

        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            double3 voxelPos;
            uint3 voxelShift = getVoxelShift(i, resolution);
            voxelPos.x = gridOrigin.x + voxelShift.x * voxelSize.x;
            voxelPos.y = gridOrigin.y + voxelShift.y * voxelSize.y;
            voxelPos.z = gridOrigin.z + voxelShift.z * voxelSize.z;

            double3 corners[8];
            corners[0] = voxelPos;
            corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
            corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
            corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
            corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
            corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
            corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
            corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

            for (int j = 0; j < 8; ++j)
                sdf[i * 8 + j] = computeSDF(corners[j]);
        });

        MC::marching_cubes(resolution, gridOrigin, voxelSize, isoVal, false, outFile, sdf);
    }*/

    return EXIT_SUCCESS;
}