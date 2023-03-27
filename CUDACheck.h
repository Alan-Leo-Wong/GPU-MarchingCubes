/*
 * @Author: Lei Wang 602289550@qq.com
 * @Date: 2023-03-18 11:18:28
 * @LastEditors: Lei Wang 602289550@qq.com
 * @LastEditTime: 2023-03-18 11:19:16
 * @FilePath: \GPUMarchingCubes\CUDACheck.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error:\n");                                        \
      fprintf(stderr, "    --File:       %s\n", __FILE__);                     \
      fprintf(stderr, "    --Line:       %d\n", __LINE__);                     \
      fprintf(stderr, "    --Error code: %d\n", error_code);                   \
      fprintf(stderr, "    --Error text: %s\n",                                \
              cudaGetErrorString(error_code));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0);

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  const cudaError_t error_code = cudaGetLastError();

  if (error_code != cudaSuccess) {
    fprintf(stderr,
            "%s(%d) : getLastCudaError() CUDA Error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(error_code),
            cudaGetErrorString(error_code));
    exit(EXIT_FAILURE);
  }
}