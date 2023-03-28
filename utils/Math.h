/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-28 13:13:26
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-28 13:16:12
 * @FilePath: \GPUMarchingCubes\utils\MathHelper.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "cuda_runtime.h"

using uint = unsigned int;

// addition
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ float3 operator+(const float3 &a, const float &b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, const float &b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

inline __host__ __device__ double3 operator+(const double3 &a,
                                             const double3 &b) {
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, const double3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ double3 operator+(const double3 &a, double &b) {
  return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, const double &b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

inline __host__ __device__ int3 operator+(const int3 a, const int3 &b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, const int3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ int3 operator+(const int3 a, const int &b) {
  return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, const int &b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

inline __host__ __device__ uint3 operator+(const uint3 a, const uint3 &b) {
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, const uint3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ uint3 operator+(const uint3 a, const uint &b) {
  return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, const uint &b) {
  a.x += b;
  a.y += b;
  a.z += b;
}