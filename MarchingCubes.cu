#include "MarchingCubes.h"
#include <cuda_device_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace MCKernel{
template <typename T> 
inline CUDA_CALLABLE_MEMBER T lerp(T v0, T v1, T t) {
  return fma(t, v1, fma(-t, v0, v0));
}

  __device__ double3 vertexLerp(const double3 &p_0, const double3 &p_1,
    const double &sdf_0, const double &sdf_1,
    const double &isoVal) {
  if (abs(isoVal - sdf_0) < 1e-6)
  return p_0;
  if (abs(isoVal - sdf_1) < 1e-6)
  return p_1;
  if (abs(sdf_1 - sdf_0) < 1e-6)
  return p_0;
  
  double t = (isoVal - sdf_0) / (sdf_1 - sdf_0);
  double3 lerp_p;
  lerp_p.x = lerp(p_0.x, p_1.x, t);
  lerp_p.y = lerp(p_0.y, p_1.y, t);
  lerp_p.z = lerp(p_0.z, p_1.z, t);
  return lerp_p;
  }
  
  __device__ double computeSDF(double3 pos) {}
  
  __device__ uint3 getVoxelShift(const uint& index, const uint3& res) {
    // TODO
    uint x = index % res.x;
    uint y = index % (res.x * res.y) / res.x;
    uint z = index / (res.x * res.y);
    return make_uint3(x, y, z);
  }
  
  /**
   * @brief 根据每个体素的 sdf 分布转为 mesh
   *
   * @param maxVerts         MC算法包含的最多的可能点数量
   * @param nValidVoxels     合理的 voxel的总数量 = res_x * res_y * res_z
   * @param voxelSize        每个 voxel 的大小
   * @param isoVal           isosurface value
   * @param d_origin         MC算法被执行的初始区域原点坐标
   * @param d_nVoxelVerts    经过 cube index 映射后每个 voxel 内应含点的数量
   * @param d_voxelCubeIndex 每个 voxel 内 sdf 分布所对应的 cube index
   * @param d_voxelSDF       每个 voxel 八个顶点的 sdf 值
   * @param d_nVertsScanned  所有合理 voxel 的点数量前缀和
   * @param d_triPoints      输出，保存实际 mesh 的所有点位置
   * @return __global__
   */
   __global__ void voxelToMeshKernel(const int maxVerts, const uint nValidVoxels,
    const double3 voxelSize, const double isoVal,
    const double3 *d_origin, int *d_nVoxelVerts,
    int *d_voxelCubeIndex, double *d_voxelSDF, 
    uint *d_nVertsScanned, double3 *d_triPoints) {
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = bid * blockDim.x + threadIdx.x;
  
  if (tid < nValidVoxels) {
  uint3 voxelShift = getVoxelShift(tid);
  double3 voxelPos; // voxel 原点坐标
  
  voxelPos.x = voxelShift.x * voxelSize.x;
  voxelPos.y = voxelShift.y * voxelSize.y;
  voxelPos.z = voxelShift.z * voxelSize.z;
  voxelPos += (*d_origin);
  
  uint cubeIndex = d_voxelCubeIndex[tid];
  double sdf[8];
  for (int i = 0; i < 8; ++i)
  sdf[i] = d_voxelSDF[tid * 8 + i];
  
  double3 corners[8];
  corners[0] = voxelPos;
  corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
  corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
  corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
  corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
  corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
  corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
  corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);
  
  // 预防线程束分化，12条边全都计算一次插值点，反正最后三角形排列方式也是由 cube index 决定
  double3 triVerts[12];
  triVerts[0] = vertexLerp(corners[0], corners[1], sdf[0], sdf[1], isoVal);
  triVerts[1] = vertexLerp(corners[1], corners[2], sdf[1], sdf[2], isoVal);
  triVerts[2] = vertexLerp(corners[2], corners[3], sdf[2], sdf[3], isoVal);
  triVerts[3] = vertexLerp(corners[3], corners[0], sdf[3], sdf[0], isoVal);
  
  triVerts[4] = vertexLerp(corners[4], corners[5], sdf[4], sdf[5], isoVal);
  triVerts[5] = vertexLerp(corners[5], corners[6], sdf[5], sdf[6], isoVal);
  triVerts[6] = vertexLerp(corners[6], corners[7], sdf[6], sdf[7], isoVal);
  triVerts[7] = vertexLerp(corners[7], corners[4], sdf[7], sdf[4], isoVal);
  
  triVerts[8] = vertexLerp(corners[0], corners[4], sdf[0], sdf[4], isoVal);
  triVerts[9] = vertexLerp(corners[1], corners[5], sdf[1], sdf[5], isoVal);
  triVerts[10] = vertexLerp(corners[2], corners[6], sdf[2], sdf[6], isoVal);
  triVerts[11] = vertexLerp(corners[3], corners[7], sdf[3], sdf[7], isoVal);
  
  int nVerts = tex1Dfetch<int>(nVertsTex, cubeIndex);
  
  for (int i = 0; i < nVerts; i += 3) {
  uint triPosIndex = d_nVertsScanned[tid] + i;
  
  double3 triangle[3];
  
  int edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i);
  triangle[0] = triVerts[edgeIndex];
  
  edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i + 1);
  triangle[1] = triVerts[edgeIndex];
  
  edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i + 2);
  triangle[2] = triVerts[edgeIndex];
  
  if (triPosIndex < maxVerts - 3) {
  d_triPoints[triPosIndex] = triangle[0];
  d_triPoints[triPosIndex + 1] = triangle[1];
  d_triPoints[triPosIndex + 2] = triangle[2];
  }
  }
  }
  }
  
  /**
   * @brief 计算每个体素的sdf值和确定分布情况
   *
   * @param nVoxels          voxel的总数量 = res_x * res_y * res_z
   * @param voxelSize        每个 voxel 的大小
   * @param isoVal           isosurface value
   * @param d_origin         MC算法被执行的初始区域原点坐标
   * @param d_nVoxelVerts    经过 cube index 映射后每个 voxel 内应含点的数量
   * @param d_nVoxelTris     经过 cube index 映射后每个 voxel 内应含三角形的数量
   * @param d_VoxelCubeIndex 每个 voxel 内 sdf 分布所对应的 cube index
   * @param d_voxelSDF       每个 voxel 八个顶点的 sdf 值
   * @param d_isValidVoxel   存储每个 voxel 的 cube index
   */
   __global__ void determineVoxelKernel(
    const uint nVoxels, const double isoVal, const double3* d_voxelSize,
    const double3 *d_origin, int *d_nVoxelVerts, int *d_nVoxelTris,
    int *d_voxelCubeIndex, double *d_voxelSDF, int *d_isValidVoxel) {
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = bid * blockDim.x + threadIdx.x;
  
  if (tid < nVoxels) {
    uint3 voxelShift = getVoxelShift(tid);
    double3 origin = *d_origin;
    double3 voxelPos; // voxel 原点坐标
  
    voxelPos.x = origin.x + voxelShift.x * d_voxelSize.x;
    voxelPos.y = origin.y + voxelShift.y * d_voxelSize.y;
    voxelPos.z = origin.z + voxelShift.z * d_voxelSize.z;
  
    double3 corners[8];
    corners[0] = voxelPos;
    corners[1] = voxelPos + make_double3(0, d_voxelSize.y, 0);
    corners[2] = voxelPos + make_double3(d_voxelSize.x, d_voxelSize.y, 0);
    corners[3] = voxelPos + make_double3(d_voxelSize.x, 0, 0);
    corners[4] = voxelPos + make_double3(0, 0, d_voxelSize.z);
    corners[5] = voxelPos + make_double3(0, d_voxelSize.y, d_voxelSize.z);
    corners[6] = voxelPos + make_double3(d_voxelSize.x, d_voxelSize.y, d_voxelSize.z);
    corners[7] = voxelPos + make_double3(d_voxelSize.x, 0, d_voxelSize.z);
  
    double sdf[8];
    for (int i = 0; i < 8; ++i) {
      sdf[i] = computeSDF(corners[i]);
      d_voxelSDF[tid * 8 + i] = sdf[i];
    }
  
    int cubeIndex = 0;
    cubeIndex = (uint(sdf[0] < isoVal)) | (uint(sdf[1] < isoVal) << 1) |
                (uint(sdf[2] < isoVal) << 2) | (uint(sdf[3] < isoVal) << 3) |
                (uint(sdf[4] < isoVal) << 4) | (uint(sdf[5] < isoVal) << 5) |
                (uint(sdf[6] < isoVal) << 6) | (uint(sdf[7] < isoVal) << 7);
  
    int nVerts = tex1Dfetch<int>(nVertsTex, cubeIndex);
    d_nVoxelVerts[tid] = nVerts;
    d_isValidVoxel[tid] = nVerts > 0;
    d_voxelCubeIndex[tid] = cubeIndex;
  }
  }
}

namespace MC {
  // host
  {
    uint *h_nVoxelVerts = nullptr;
    uint *h_nVoxelVertsScan = nullptr;
  
    uint *h_nValidVoxels = nullptr;
    uint *h_nValidVoxelsScan = nullptr;
  }

  // device
  {
    uint *d_nVoxelVerts = nullptr;
    uint *d_nVoxelVertsScan = nullptr;
  
    uint *d_nValidVoxels = nullptr;
    uint *d_nValidVoxelsScan = nullptr;
  
    double3* d_gridOrigin = nullptr;
    double3* d_voxelSize = nullptr;
  
    double* d_voxelSDF = nullptr;

    int* d_triTable = nullptr;
    int* d_nVertsTable = nullptr;

    // textures containing look-up tables
    cudaTextureObject_t triTex;
    cudaTextureObject_t nVertsTex;

    double3* d_triPoints = nullptr;
  }
}

inline void MC::d_thrustExclusiveScan(const uint& nElems, uint* input, uint* output)
{
  thrust::exclusive_scan(thrust::device_ptr<uint>(input), 
                         thrust::device_ptr<uint>(input + nElems), 
                         thrust::device_ptr<uint>(output);
}

inline void MC::setTextureObject(const uint& srcSizeInBytes, int* srcDev, cudaTextureObject_t* texObj)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaResourceDesc texRes, texDesc;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  memset(&texDesc, 0, sizeof(cudaTextureDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = srcDev;
  texRes.res.linear.sizeInBytes = srcSizeInBytes;
  texRes.res.linear.desc = channelDesc;

  texDesc.normalizedCoords = false;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.readMode = cudaReadModeElementType;

  CUDA_CHECK(cudaCreateTextureObject(texObj, &texRes, &texDesc, nullptr));
}

inline void MC::initResources(cosnt uint& nVoxels, const double3& gridOrigin, const double3& voxelSize, const uint& maxVerts)
{
  // host
  {
    h_nVoxelVerts = malloc(sizeof(uint) * nVoxels);
    h_nVoxelVertsScan = malloc(sizeof(uint) * nVoxels);

    h_nValidVoxels = malloc(sizeof(uint) * nVoxels);
    h_nValidVoxelsScan = malloc(sizeof(uint) * nVoxels);  
  }
  
  // device
  {
    CUDA_CHECK(cudaMalloc(d_nVoxelVerts), sizeof(uint) * nVoxels);
    CUDA_CHECK(cudaMalloc(d_nVoxelVertsScan), sizeof(uint) * nVoxels);
  
    CUDA_CHECK(cudaMalloc(d_nValidVoxels), sizeof(uint) * nVoxels);
    CUDA_CHECK(cudaMalloc(d_nValidVoxelsScan), sizeof(uint) * nVoxels);
  
    CUDA_CHECK(cudaMalloc(d_gridOrigin, sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(d_gridOrigin, gridOrigin, sizeof(double3), cudaMemcpyHostToDevice));
  
    CUDA_CHECK(cudaMalloc(d_voxelSize, sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(d_voxelSize, voxelSize, sizeof(double3), cudaMemcpyHostToDevice));
  
    CUDA_CHECK(cudaMalloc(d_voxelSDF, sizeof(double) * nVoxels * 8));

    CUDA_CHEKC(cudaMalloc(d_triTable, sizeof(int) * 256 * 16));
    CUDA_CHECK(cudaMemcpy(d_triTable, triTable, sizeof(int) * 256 * 16, cudaMemcpyHostToDevice));
  
    CUDA_CHEKC(cudaMalloc(d_nVertsTable, sizeof(int) * 256));
    CUDA_CHECK(cudaMemcpy(d_nVertsTable, nVertsTable, sizeof(int) * 256, cudaMemcpyHostToDevice));
  
    // texture
    setTextureObject(256 * 16 * sizeof(int), d_triTable, triTex);
    setTextureObject(256 * sizeof(int), d_nVertsTable, nVertsTex);

    CUDA_CHECK(cudaMalloc(d_triPoints, sizeof(double3) * maxVerts));
  }
}

inline void MC::freeResources()
{
    // host
    {
      free(h_nVoxelVerts);
      free(h_nVoxelVertsScan);

      free(h_nValidVoxels);
      free(h_nValidVoxelsScan);
    }

    // device
    {
      CUDA_CHECK(cudaFree(d_nVoxelVerts));
      CUDA_CHECK(cudaFree(d_nVoxelVertsScan);)

      CUDA_CHECK(cudaFree(d_nValidVoxels));
      CUDA_CHECK(cudaFree(d_nValidVoxelsScan));

      CUDA_CHECK(cudaFree(d_gridOrigin));
      CUDA_CHECK(cudaFree(d_voxelSize));

      CUDA_CHECK(cudaFree(d_voxelSDF));

      CUDA_CHECK(cudaFree(d_triTable));
      CUDA_CHECK(cudaFree(d_nVertsTable));

      // texture object
      CUDA_CHECK(cudaDestroyTextureObject(triTex));
      CUDA_CHECK(cudaDestroyTextureObject(nVertsTex));
    }
}

inline void MC::launch_determineVoxelKernel(const uint &nVoxels, 
                                            const double3& voxelSize, 
                                            const double& isoVal,
                                            const uint& maxVerts) {
    CUDA_CHECK(cudaMalloc((void **)&d_nVoxelVerts, sizeof(int) * nVoxels));
    CUDA_CHECK(cudaMalloc((void **)&d_voxelVertsScan, sizeof(int) * nVoxels));

    dim3 nThreads(256, 1, 1);
    dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
    if (nBlocks.x > 65535) {
      nBlocks.y = nBlocks.x / 32768;
      nBlocks.x = 32768;
    }

    MCKernel::determineVoxelKernel<<<nBlocks, nThreads>>>();
    getLastCudaErrors("Kernel 'determineVoxelKernel' error!\n");

} 

inline void MC::launch_voxelToMeshKernel(const uint &nVoxels)
{

}

void MC::marching_cubes(int argc, char **argv) {
    uint3 grid = make_uint3(20, 20, 20); // resolution
    uint nVoxels = grid.x * grid.y * grid.z;
    uint maxVerts = nVoxels * 20;
  
    float3 gridOrigin = make_float3(0, 0, 0);   // origin coordinate of grid
    float3 gridWidth = make_float3(20, 20, 20); // with of grid
    float3 voxelSize = make_float3(gridWidth.x / grid.x, gridWidth.y / grid.y,
                                   gridWidth.z / grid.z);
  
    initResources(nVoxels, gridOrigin, voxelSize, maxVerts);

    launch_determineVoxelKernel(nVoxels);

    freeResources();
  }
