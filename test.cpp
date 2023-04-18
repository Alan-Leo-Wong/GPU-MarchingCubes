/*
 * @Author: Lei Wang 602289550@qq.com
 * @Date: 2023-03-18 11:14:25
 * @LastEditors: WangLei
 * @LastEditTime: 2023-04-18 23:03:50
 * @FilePath: \GPUMarchingCubes\test.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "MarchingCubes.h"
#include <iostream>
#include <stdlib.h>

int main(int argc, char **argv) {
  uint3 resolution = make_uint3(20, 20, 20);
  double3 gridOrigin = make_double3(-1, -1, -1); // origin coordinate of grid
  double3 gridWidth = make_double3(2, 2, 2);     // width of grid
  double isoVal = .0;
  const std::string filename = ".\\test\\sphere.obj";
  MC::marching_cubes(resolution, gridOrigin, gridWidth, isoVal, true, filename);

  //   std::vector<double> sdf(resolution.x * resolution.y * resolution.z * 8,
  //   0); sdf[0] = -0.5; sdf[1] = 0.5; sdf[2] = -0.5; sdf[3] = 0.5; sdf[4] =
  //   -0.5; sdf[5] = 0.5; sdf[6] = -0.5; sdf[7] = -0.5;
  //   MC::marching_cubes(resolution, gridOrigin, gridWidth, isoVal, false,
  //   filename,
  //                      sdf);

  return EXIT_SUCCESS;
}