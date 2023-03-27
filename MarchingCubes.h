/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-18 11:31:45
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-27 19:05:26
 * @FilePath: \GPUMarchingCubes\MarchingCubes.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "CUDACheck.h"
#include "CUDAMacro.h"

namespace MC {
using uint = unsigned int;

void launch_determineVoxelKernel(const uint &nVoxels);

void launch_voxelToMeshKernel(const uint &nVoxels);

void marching_cubes(int argc, char **argv);
} // namespace MC
