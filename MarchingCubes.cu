#include "Define.h"
#include "LookTable.h"
#include "MarchingCubes.h"
#include "utils\String.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <texture_types.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector_functions.h>
#include <vector_types.h>

inline __device__ double3 MCKernel::vertexLerp(const double3 &p_0,
                                               const double3 &p_1,
                                               const double &sdf_0,
                                               const double &sdf_1,
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

inline __device__ double MCKernel::computeSDF(double3 pos) {}

inline __device__ uint3 MCKernel::getVoxelShift(const uint &index,
                                                const uint3 &d_res) {
  // TODO
  uint x = index % d_res.x;
  uint y = index % (d_res.x * d_res.y) / d_res.x;
  uint z = index / (d_res.x * d_res.y);
  return make_uint3(x, y, z);
}

/**
 * @brief 计算每个体素的sdf值和确定分布情况
 *
 * @param nVoxels          voxel的总数量 = res_x * res_y * res_z
 * @param voxelSize        每个 voxel 的大小
 * @param d_isoVal           isosurface value
 * @param d_origin         MC算法被执行的初始区域原点坐标
 * @param d_nVoxelVerts    经过 cube index 映射后每个 voxel 内应含点的数量
 * @param d_VoxelCubeIndex 每个 voxel 内 sdf 分布所对应的 cube index
 * @param d_voxelSDF       每个 voxel 八个顶点的 sdf 值
 * @param d_isValidVoxel   存储每个 voxel 的 cube index
 */
__global__ void MCKernel::determineVoxelKernel(
    const uint nVoxels, const double *d_isoVal, const double3 *d_voxelSize,
    const double3 *d_origin, const uint3 *d_res,
    const cudaTextureObject_t nVertsTex, uint *d_nVoxelVerts,
    uint *d_voxelCubeIndex, double *d_voxelSDF, uint *d_isValidVoxel) {
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = bid * blockDim.x + threadIdx.x;

  if (tid < nVoxels) {
    double isoVal = *d_isoVal;

    uint3 voxelShift = getVoxelShift(tid, *d_res);
    double3 origin = *d_origin;
    double3 voxelSize = *d_voxelSize;
    double3 voxelPos; // voxel 原点坐标

    voxelPos.x = origin.x + voxelShift.x * voxelSize.x;
    voxelPos.y = origin.y + voxelShift.y * voxelSize.y;
    voxelPos.z = origin.z + voxelShift.z * voxelSize.z;

    double3 corners[8];
    corners[0] = voxelPos;
    corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
    corners[2] = voxelPos + make_double3(voxelSize.x, d_voxelSize.y, 0);
    corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
    corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
    corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
    corners[6] =
        voxelPos + make_double3(d_voxelSize.x, voxelSize.y, voxelSize.z);
    corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

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

/**
 * @brief 根据每个体素的 sdf 分布转为 mesh
 *
 * @param maxVerts         MC算法包含的最多的可能点数量
 * @param nValidVoxels     合理的 voxel的总数量 = res_x * res_y * res_z
 * @param voxelSize        每个 voxel 的大小
 * @param d_isoVal           isosurface value
 * @param d_origin         MC算法被执行的初始区域原点坐标
 * @param d_nVoxelVerts    经过 cube index 映射后每个 voxel 内应含点的数量
 * @param d_voxelCubeIndex 每个 voxel 内 sdf 分布所对应的 cube index
 * @param d_voxelSDF       每个 voxel 八个顶点的 sdf 值
 * @param d_nVertsScanned  所有合理 voxel 的点数量前缀和
 * @param d_triPoints      输出，保存实际 mesh 的所有点位置
 */
__global__ void MCKernel::voxelToMeshKernel(
    const uint nValidVoxels, const int maxVerts, const double *d_isoVal,
    const double3 *d_voxelSize, const double3 *d_origin, const uint3 *d_res,
    const cudaTextureObject_t nVertsTex, const cudaTextureObject_t triTex,
    uint *d_voxelCubeIndex, double *d_voxelSDF, uint *d_nVertsScanned,
    double3 *d_triPoints) {
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = bid * blockDim.x + threadIdx.x;

  if (tid < nValidVoxels) {
    double isoVal = *d_isoVal;

    uint3 voxelShift = getVoxelShift(tid, *d_res);
    double3 voxelPos; // voxel 原点坐标
    double3 voxelSize = *d_voxelSize;

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

    // 预防线程束分化，12条边全都计算一次插值点，反正最后三角形排列方式也是由
    // cube index 决定
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

namespace MC {
// host
namespace {
uint allTriVertices = 0, nValidVoxels = 0;

uint *h_nVoxelVerts = nullptr;
uint *h_nVoxelVertsScan = nullptr;

uint *h_nValidVoxels = nullptr;
uint *h_nValidVoxelsScan = nullptr;

double3 *h_triPoints = nullptr; // output
} // namespace

// device
namespace {
uint3 *d_res = nullptr;
double *d_isoVal = nullptr;

uint *d_nVoxelVerts = nullptr;
uint *d_nVoxelVertsScan = nullptr;

uint *d_isValidVoxels = nullptr;
uint *d_nValidVoxelsScan = nullptr;

double3 *d_gridOrigin = nullptr;
double3 *d_voxelSize = nullptr;

double *d_voxelSDF = nullptr;
uint *d_voxelCubeIndex = nullptr;

int *d_triTable = nullptr;
int *d_nVertsTable = nullptr;

// textures containing look-up tables
cudaTextureObject_t triTex;
cudaTextureObject_t nVertsTex;

double3 *d_triPoints = nullptr; // output
} // namespace
} // namespace MC

inline void MC::d_thrustExclusiveScan(const uint &nElems, uint *input,
                                      uint *output) {
  thrust::exclusive_scan(thrust::device_ptr<uint>(input),
                         thrust::device_ptr<uint>(input + nElems),
                         thrust::device_ptr<uint>(output));
}

inline void MC::setTextureObject(const uint &srcSizeInBytes, int *srcDev,
                                 cudaTextureObject_t *texObj) {
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaResourceDesc texRes;
  cudaTextureDesc texDesc;
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

inline void MC::initResources(const uint3 &resolution, const uint &nVoxels,
                              const double &isoVal, const double3 &gridOrigin,
                              const double3 &voxelSize, const uint &maxVerts) {
  // host
  {
    h_nVoxelVerts = (uint *)malloc(sizeof(uint) * nVoxels);
    h_nVoxelVertsScan = (uint *)malloc(sizeof(uint) * nVoxels);

    h_nValidVoxels = (uint *)malloc(sizeof(uint) * nVoxels);
    h_nValidVoxelsScan = (uint *)malloc(sizeof(uint) * nVoxels);

    h_triPoints = (double3 *)malloc(sizeof(double3) * maxVerts);
  }

  // device
  {
    CUDA_CHECK(cudaMalloc((void **)&d_res, sizeof(uint3)));
    CUDA_CHECK(
        cudaMemcpy(d_res, &resolution, sizeof(uint3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_isoVal, sizeof(double)));
    CUDA_CHECK(
        cudaMemcpy(d_isoVal, &isoVal, sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_nVoxelVerts, sizeof(uint) * nVoxels));
    CUDA_CHECK(cudaMalloc((void **)&d_nVoxelVertsScan, sizeof(uint) * nVoxels));

    CUDA_CHECK(cudaMalloc((void **)&d_isValidVoxels, sizeof(uint) * nVoxels));
    CUDA_CHECK(
        cudaMalloc((void **)&d_nValidVoxelsScan, sizeof(uint) * nVoxels));

    CUDA_CHECK(cudaMalloc((void **)&d_gridOrigin, sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(d_gridOrigin, &gridOrigin, sizeof(double3),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_voxelSize, sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(d_voxelSize, &voxelSize, sizeof(double3),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_voxelSDF, sizeof(double) * nVoxels * 8));
    CUDA_CHECK(cudaMalloc((void **)&d_voxelCubeIndex, sizeof(uint) * nVoxels));

    CUDA_CHECK(cudaMalloc((void **)&d_triTable, sizeof(int) * 256 * 16));
    CUDA_CHECK(cudaMemcpy(d_triTable, triTable, sizeof(int) * 256 * 16,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_nVertsTable, sizeof(int) * 256));
    CUDA_CHECK(cudaMemcpy(d_nVertsTable, nVertsTable, sizeof(int) * 256,
                          cudaMemcpyHostToDevice));

    // texture
    setTextureObject(256 * 16 * sizeof(int), d_triTable, &triTex);
    setTextureObject(256 * sizeof(int), d_nVertsTable, &nVertsTex);

    CUDA_CHECK(cudaMalloc((void **)&d_triPoints, sizeof(double3) * maxVerts));
  }
}

inline void MC::freeResources() {
  // host
  {
    free(h_nVoxelVerts);
    free(h_nVoxelVertsScan);

    free(h_nValidVoxels);
    free(h_nValidVoxelsScan);

    free(h_triPoints);
  }

  // device
  {
    CUDA_CHECK(cudaFree(d_res));

    CUDA_CHECK(cudaFree(d_nVoxelVerts));
    CUDA_CHECK(cudaFree(d_nVoxelVertsScan);)

    CUDA_CHECK(cudaFree(d_isValidVoxels));
    CUDA_CHECK(cudaFree(d_nValidVoxelsScan));

    CUDA_CHECK(cudaFree(d_gridOrigin));
    CUDA_CHECK(cudaFree(d_voxelSize));

    CUDA_CHECK(cudaFree(d_voxelSDF));
    CUDA_CHECK(cudaFree(d_voxelCubeIndex));

    CUDA_CHECK(cudaFree(d_triTable));
    CUDA_CHECK(cudaFree(d_nVertsTable));

    // texture object
    CUDA_CHECK(cudaDestroyTextureObject(triTex));
    CUDA_CHECK(cudaDestroyTextureObject(nVertsTex));

    CUDA_CHECK(cudaFree(d_triPoints));
  }
}

inline void MC::launch_determineVoxelKernel(const uint &nVoxels,
                                            const double &isoVal,
                                            const uint &maxVerts) {
  dim3 nThreads(NTHREADS, 1, 1);
  dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
  while (nBlocks.x > 65535) {
    nBlocks.x /= 2;
    nBlocks.y *= 2;
  }

  MCKernel::determineVoxelKernel<<<nBlocks, nThreads>>>(
      nVoxels, d_isoVal, d_voxelSize, d_gridOrigin, d_res, nVertsTex,
      d_nVoxelVerts, d_voxelCubeIndex, d_voxelSDF, d_isValidVoxels);
  getLastCudaError("Kernel: 'determineVoxelKernel' error!\n");

  d_thrustExclusiveScan(nVoxels, d_nVoxelVerts, d_nVoxelVertsScan);
  d_thrustExclusiveScan(nVoxels, d_isValidVoxels, d_nValidVoxelsScan);

  uint lastElement, lastScanElement;
  CUDA_CHECK(cudaMemcpy(&lastElement, d_isValidVoxels + nVoxels, sizeof(uint),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nValidVoxelsScan + nVoxels,
                        sizeof(uint), cudaMemcpyDeviceToHost));
  nValidVoxels = lastElement + lastScanElement;
  if (nValidVoxels == 0)
    return;

  CUDA_CHECK(cudaMemcpy(&lastElement, d_nVoxelVerts + nVoxels, sizeof(uint),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nVoxelVerts + nVoxels, sizeof(uint),
                        cudaMemcpyDeviceToHost));
  allTriVertices = lastElement + lastScanElement;

  CUDA_CHECK(cudaMemcpy(&h_nVoxelVerts, d_nVoxelVerts, sizeof(uint) * nVoxels,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_nVoxelVertsScan, d_nVoxelVertsScan,
                        sizeof(uint) * nVoxels, cudaMemcpyDeviceToHost));
}

inline void MC::launch_voxelToMeshKernel(const uint &maxVerts) {
  dim3 nThreads(NTHREADS, 1, 1);
  dim3 nBlocks((nValidVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
  while (nBlocks.x > 65535) {
    nBlocks.x /= 2;
    nBlocks.y *= 2;
  }

  MCKernel::voxelToMeshKernel<<<nBlocks, nThreads>>>(
      nValidVoxels, maxVerts, d_isoVal, d_voxelSize, d_gridOrigin, d_res,
      nVertsTex, triTex, d_voxelCubeIndex, d_voxelSDF, d_nVoxelVertsScan,
      d_triPoints);
}

void MC::writeToOBJFile(const std::string &filename) {
  checkDir(filename);
  std::ofstream out(filename);
  if (!out) {
    fprintf(stderr, "IO Error: File %s could not be opened!", filename.c_str());
    return;
  }

  for (int i = 0; i < allTriVertices; i += 3) {
    if (h_nVoxelVerts[i] == 0)
      continue;
    const int faceIdx = h_nVoxelVertsScan[i] / 3;

    out << "v " << h_triPoints[i].x << ' ' << h_triPoints[i].y << ' '
        << h_triPoints[i].z << '\n';
    out << "v " << h_triPoints[i + 1].x << ' ' << h_triPoints[i + 1].y << ' '
        << h_triPoints[i + 1].z << '\n';
    out << "v " << h_triPoints[i + 2].x << ' ' << h_triPoints[i + 2].y << ' '
        << h_triPoints[i + 2].z << '\n';

    out << "f " << faceIdx << faceIdx + 1 << faceIdx + 2 << '\n';
  }

  out.close();
}

void MC::marching_cubes(int argc, char **argv) {
  uint3 resolution = make_uint3(20, 20, 20); // resolution
  uint nVoxels = resolution.x * resolution.y * resolution.z;
  uint maxVerts = nVoxels * 18;
  double isoVal = .0;
  const string filename = "";

  double3 gridOrigin = make_double3(0, 0, 0);   // origin coordinate of grid
  double3 gridWidth = make_double3(20, 20, 20); // with of grid
  double3 voxelSize =
      make_double3(gridWidth.x / resolution.x, gridWidth.y / resolution.y,
                   gridWidth.z / resolution.z);

  initResources(resolution, nVoxels, isoVal, gridOrigin, voxelSize, maxVerts);

  launch_determineVoxelKernel(nVoxels, isoVal, maxVerts);
  if (allTriVertices == 0) {
    printf("There is no vertices...\n");
    return;
  }

  launch_voxelToMeshKernel(maxVerts);

  freeResources();

  writeToOBJFile(filename);
}
