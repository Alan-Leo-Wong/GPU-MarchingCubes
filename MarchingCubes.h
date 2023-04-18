/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-18 11:31:45
 * @LastEditors: WangLei
 * @LastEditTime: 2023-04-18 22:21:10
 * @FilePath: \GPUMarchingCubes\MarchingCubes.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "CUDACheck.h"
#include "utils\Math.h"
#include <vector>

namespace MCKernel {
template <typename T> inline __host__ __device__ T lerp(T v0, T v1, T t) {
  return fma(t, v1, fma(-t, v0, v0));
}

__device__ double3 vertexLerp(const double3 &p_0, const double3 &p_1,
                              const double &sdf_0, const double &sdf_1,
                              const double &isoVal);

__device__ double computeSDF(double3 pos);

__device__ uint3 getVoxelShift(const uint &index, const uint3 &d_res);

__global__ void
determineVoxelKernel(const uint nVoxels, const double *d_isoVal,
                     const double3 *d_voxelSize, const double3 *d_origin,
                     const uint3 *d_res, const cudaTextureObject_t nVertsTex,
                     uint *d_nVoxelVerts, uint *d_voxelCubeIndex,
                     double *d_voxelSDF, uint *d_isValidVoxel);

__global__ void compactVoxels(const uint nVoxels, const uint *d_isValidVoxel,
                              const uint *d_nValidVoxelsScan,
                              uint *d_compactedVoxelArray);

__global__ void voxelToMeshKernel(
    const uint nValidVoxels, const int maxVerts, const double *d_isoVal,
    const double3 *d_voxelSize, const double3 *d_origin, const uint3 *d_res,
    const uint *d_compactedVoxelArray, const cudaTextureObject_t nVertsTex,
    const cudaTextureObject_t triTex, uint *d_voxelCubeIndex,
    double *d_voxelSDF, uint *d_nVertsScanned, double3 *d_triPoints);
} // namespace MCKernel

namespace MC {
void d_thrustExclusiveScan(const uint &nElems, uint *input, uint *output);

void setTextureObject(const uint &srcSizeInBytes, int *srcDev,
                      cudaTextureObject_t *texObj);

void initResources(const bool &sdfFlag, const uint3 &resolution,
                   const uint &nVoxels, const double &isoVal,
                   const double3 &gridOrigin, const double3 &voxelSize,
                   const uint &maxVerts, const std::vector<double> h_voxelSDF);

void freeResources();

void launch_determineVoxelKernel(const uint &nVoxels, const double &isoVal,
                                 const uint &maxVerts);

void launch_compactVoxelsKernel(const int &nVoxels);

void launch_voxelToMeshKernel(const uint &maxVerts, const uint &nVoxels);

void writeToOBJFile(const std::string &filename);

void marching_cubes(
    const uint3 &resolution, const double3 &gridOrigin,
    const double3 &gridWidth, const double &isoVal, const bool &sdfFlag,
    const std::string &filename,
    const std::vector<double> &h_voxelSDF = std::vector<double>());
} // namespace MC
