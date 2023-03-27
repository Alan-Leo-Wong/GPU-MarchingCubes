/*
 * @Author: Lei Wang 602289550@qq.com
 * @Date: 2023-03-18 11:14:25
 * @LastEditors: Lei Wang 602289550@qq.com
 * @LastEditTime: 2023-03-18 11:30:52
 * @FilePath: \GPUMarchingCubes\test.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "MarchingCubes.h"
#include "vector_functions.h"
#include "vector_types.h"
#include <iostream>
#include <stdlib.h>

uint nVerts = 0; // number of vertices

int main(int argc, char **argv) {
  marching_cubes(argc, argv);

  return EXIT_SUCCESS;
}