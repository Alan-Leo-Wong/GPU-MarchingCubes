/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-18 11:31:45
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-27 19:58:27
 * @FilePath: \GPUMarchingCubes\MarchingCubes.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "CUDACheck.h"
#include "CUDAMacro.h"

using uint = unsigned int;

namespace MCKernel {
template <typename T> inline CUDA_CALLABLE_MEMBER T lerp(T v0, T v1, T t) {
  return fma(t, v1, fma(-t, v0, v0));
}

__device__ double3 vertexLerp(const double3 &p_0, const double3 &p_1,
                              const double &sdf_0, const double &sdf_1,
                              const double &isoVal);

__device__ double computeSDF(double3 pos);

__device__ uint3 getVoxelShift(const uint &index, const uint3 &d_res);

__global__ void determineVoxelKernel(
    const uint nVoxels, const double isoVal, const double3 *d_voxelSize,
    const double3 *d_origin, const uint3 *d_res,
    const cudaTextureObject_t nVertsTex, int *d_nVoxelVerts, int *d_nVoxelTris,
    int *d_voxelCubeIndex, double *d_voxelSDF, int *d_isValidVoxel);

__global__ void voxelToMeshKernel(
    const int maxVerts, const double isoVal, const double3 *d_voxelSize,
    const double3 *d_origin, const uint3 *d_res, const uint nValidVoxels,
    const cudaTextureObject_t nVertsTex, const cudaTextureObject_t triTex,
    int *d_nVoxelVerts, int *d_voxelCubeIndex, double *d_voxelSDF,
    uint *d_nVertsScanned, double3 *d_triPoints);
} // namespace MCKernel

namespace MC {
void d_thrustExclusiveScan(const uint &nElems, uint *input, uint *output);

void setTextureObject(const uint &srcSizeInBytes, int *srcDev,
                      cudaTextureObject_t *texObj);

void initResources(const uint3 &resolution, const uint &nVoxels,
                   const double3 &gridOrigin, const double3 &voxelSize,
                   const uint &maxVerts);

void freeResources();

void launch_determineVoxelKernel(const uint &nVoxels);

void launch_voxelToMeshKernel(const uint &nVoxels);

void marching_cubes(int argc, char **argv);
} // namespace MC
