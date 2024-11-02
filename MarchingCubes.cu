/*
 * @Author: Lei Wang leiw1006@gmail.com
 * @Date: 2023-03-18 11:32:15
 * @LastEditors: Lei Wang
 * @LastEditTime: 2023-04-18 23:01:05
 * @FilePath: \GPUMarchingCubes\MarchingCubes.h
 */
#include "Define.h"
#include "LookTable.h"
#include "MarchingCubes.h"
#include "utils\String.h"
#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <fstream>
#include <functional>
#include <texture_types.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>

__device__ double3 MCKernel::vertexLerp(const double3 &p_0,
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

__device__ double MCKernel::computeSDF(double3 pos) {
    // here is a sphere function example
    return pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
}

__device__ uint3 MCKernel::getVoxelShift(const uint &index,
                                         const uint3 &d_res) {
    // TODO
    uint x = index % d_res.x;
    uint y = index % (d_res.x * d_res.y) / d_res.x;
    uint z = index / (d_res.x * d_res.y);
    return make_uint3(x, y, z);
}

__device__ bool isNeedComputeSDF = true;

/**
 * @brief Calculates the SDF value for each voxel and determines its distribution.
 *
 * @param nVoxels          Total number of voxels = res_x * res_y * res_z
 * @param voxelSize        Size of each voxel
 * @param d_isoVal         Isosurface value
 * @param d_origin         Origin coordinates of the initial region where the MC algorithm is executed
 * @param d_res            Resolution
 * @param d_nVoxelVerts    Number of points that should be contained in each voxel after cube index mapping
 * @param d_VoxelCubeIndex Cube index corresponding to the SDF distribution within each voxel
 * @param d_voxelSDF       SDF values of the eight vertices of each voxel
 * @param d_isValidVoxel   Determines whether each voxel is a valid voxel
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
        double3 voxelPos; // the origin coordinate of the voxel

        voxelPos.x = origin.x + voxelShift.x * voxelSize.x;
        voxelPos.y = origin.y + voxelShift.y * voxelSize.y;
        voxelPos.z = origin.z + voxelShift.z * voxelSize.z;

        double3 corners[8];
        corners[0] = voxelPos;
        corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
        corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
        corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
        corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
        corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
        corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
        corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

        double sdf[8];
        if (isNeedComputeSDF) {
            for (int i = 0; i < 8; ++i) {
                sdf[i] = computeSDF(corners[i]);
                d_voxelSDF[tid * 8 + i] = sdf[i];
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                sdf[i] = d_voxelSDF[tid * 8 + i];
#ifndef NDEBUG
                if (tid == 0) {
                  printf("sdf = %lf\n", sdf[i]);
                }
#endif
            }
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
 * @brief Compact voxel array
 *
 * @param nVoxels               Total number of voxels = res_x * res_y * res_z
 * @param d_isValidVoxel        Determines whether each voxel is a valid voxel
 * @param d_nValidVoxelsScan    Exclusive sum of d_isValidVoxel
 * @param d_compactedVoxelArray Output
 */
__global__ void MCKernel::compactVoxels(const uint nVoxels,
                                        const uint *d_isValidVoxel,
                                        const uint *d_nValidVoxelsScan,
                                        uint *d_compactedVoxelArray) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;

    if (tid < nVoxels && d_isValidVoxel[tid])
        d_compactedVoxelArray[d_nValidVoxelsScan[tid]] = tid;
}

/**
 * @brief Converts the SDF distribution of each voxel into a mesh
 *
 * @param maxVerts              Maximum possible number of points included by the MC algorithm
 * @param nValidVoxels          Total number of valid voxels = res_x * res_y * res_z
 * @param voxelSize             Size of each voxel
 * @param d_isoVal              Isosurface value
 * @param d_origin              Origin coordinates of the initial region where the MC algorithm is executed
 * @param d_res                 Resolution
 * @param d_compactedVoxelArray Array of voxels with invalid entries removed
 * @param d_nVoxelVerts         Number of points that should be contained in each voxel after cube index mapping
 * @param d_voxelCubeIndex      Cube index corresponding to the SDF distribution within each voxel
 * @param d_voxelSDF            SDF values of the eight vertices of each voxel
 * @param d_nVertsScanned       Prefix sum of the point count in all valid voxels
 * @param d_triPoints           Output, stores the position of all points in the actual mesh
 */
__global__ void MCKernel::voxelToMeshKernel(
        const uint nValidVoxels, const int maxVerts, const double *d_isoVal,
        const double3 *d_voxelSize, const double3 *d_origin, const uint3 *d_res,
        const uint *d_compactedVoxelArray, const cudaTextureObject_t nVertsTex,
        const cudaTextureObject_t triTex, uint *d_voxelCubeIndex,
        double *d_voxelSDF, uint *d_nVertsScanned, double3 *d_triPoints) {
    uint bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint tid = bid * blockDim.x + threadIdx.x;

    if (tid < nValidVoxels) {
        uint voxelIdx = d_compactedVoxelArray[tid];

        double isoVal = *d_isoVal;

        uint3 voxelShift = getVoxelShift(voxelIdx, *d_res);
        double3 voxelPos; // voxel 原点坐标
        double3 voxelSize = *d_voxelSize;

        voxelPos.x = voxelShift.x * voxelSize.x;
        voxelPos.y = voxelShift.y * voxelSize.y;
        voxelPos.z = voxelShift.z * voxelSize.z;
        voxelPos += (*d_origin);

        uint cubeIndex = d_voxelCubeIndex[voxelIdx];
        double sdf[8];
        for (int i = 0; i < 8; ++i)
            sdf[i] = d_voxelSDF[voxelIdx * 8 + i];

        double3 corners[8];
        corners[0] = voxelPos;
        corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
        corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
        corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
        corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
        corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
        corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
        corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

        // To prevent thread divergence, calculate the interpolation points on all 12 edges once,
        // since the final triangle arrangement is also determined by the cube index.
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
            uint triPosIndex = d_nVertsScanned[voxelIdx] + i;

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

        double3 *h_triPoints = nullptr; // output
    } // namespace

    // device
    namespace {
        uint3 *d_res = nullptr;
        double *d_isoVal = nullptr;

        uint *d_nVoxelVertsArray = nullptr;
        uint *d_nVoxelVertsScan = nullptr;

        uint *d_isValidVoxelArray = nullptr;
        uint *d_nValidVoxelsScan = nullptr;

        double3 *d_gridOrigin = nullptr;
        double3 *d_voxelSize = nullptr;

        double *d_voxelSDF = nullptr;
        uint *d_voxelCubeIndex = nullptr;

        uint *d_compactedVoxelArray = nullptr;

        int *d_triTable = nullptr;
        int *d_nVertsTable = nullptr;

        // textures containing look-up tables
        cudaTextureObject_t triTex;
        cudaTextureObject_t nVertsTex;

        double3 *d_triPoints = nullptr; // output
    } // namespace
} // namespace MC

void MC::d_thrustExclusiveScan(const uint &nElems, uint *input,
                               uint *output) {
    thrust::exclusive_scan(thrust::device_ptr<uint>(input),
                           thrust::device_ptr<uint>(input + nElems),
                           thrust::device_ptr<uint>(output));
}

void MC::setTextureObject(const uint &srcSizeInBytes, int *srcDev,
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

void MC::initResources(const bool &sdfFlag, const uint3 &resolution,
                       const uint &nVoxels, const double &isoVal,
                       const double3 &gridOrigin,
                       const double3 &voxelSize, const uint &maxVerts,
                       const std::vector<double> h_voxelSDF) {
    // host
    {
        h_triPoints = (double3 *) malloc(sizeof(double3) * maxVerts);
        // printf("h_triPoints = %d\n", h_triPoints);
    }

    // device
    {
        CUDA_CHECK(cudaMalloc((void **) &d_res, sizeof(uint3)));
        CUDA_CHECK(
                cudaMemcpy(d_res, &resolution, sizeof(uint3), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &d_isoVal, sizeof(double)));
        CUDA_CHECK(
                cudaMemcpy(d_isoVal, &isoVal, sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(
                cudaMalloc((void **) &d_nVoxelVertsArray, sizeof(uint) * nVoxels));
        CUDA_CHECK(cudaMalloc((void **) &d_nVoxelVertsScan, sizeof(uint) * nVoxels));

        CUDA_CHECK(
                cudaMalloc((void **) &d_isValidVoxelArray, sizeof(uint) * nVoxels));
        CUDA_CHECK(
                cudaMalloc((void **) &d_nValidVoxelsScan, sizeof(uint) * nVoxels));

        CUDA_CHECK(cudaMalloc((void **) &d_gridOrigin, sizeof(double3)));
        CUDA_CHECK(cudaMemcpy(d_gridOrigin, &gridOrigin, sizeof(double3),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &d_voxelSize, sizeof(double3)));
        CUDA_CHECK(cudaMemcpy(d_voxelSize, &voxelSize, sizeof(double3),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &d_voxelSDF, sizeof(double) * nVoxels * 8));
        if (!sdfFlag) {
            assert(h_voxelSDF.size() >= nVoxels * 8);
            CUDA_CHECK(cudaMemcpyToSymbol(isNeedComputeSDF, &sdfFlag, sizeof(bool)));
            CUDA_CHECK(cudaMemcpy(d_voxelSDF, h_voxelSDF.data(),
                                  sizeof(double) * nVoxels * 8,
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMalloc((void **) &d_voxelCubeIndex, sizeof(uint) * nVoxels));

        CUDA_CHECK(cudaMalloc((void **) &d_triTable, sizeof(int) * 256 * 16));
        CUDA_CHECK(cudaMemcpy(d_triTable, triTable, sizeof(int) * 256 * 16,
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void **) &d_nVertsTable, sizeof(int) * 256));
        CUDA_CHECK(cudaMemcpy(d_nVertsTable, nVertsTable, sizeof(int) * 256,
                              cudaMemcpyHostToDevice));

        // texture
        setTextureObject(256 * 16 * sizeof(int), d_triTable, &triTex);
        setTextureObject(256 * sizeof(int), d_nVertsTable, &nVertsTex);

        CUDA_CHECK(cudaMalloc((void **) &d_triPoints, sizeof(double3) * maxVerts));
    }
}

void MC::freeResources() {
    // host
    { free(h_triPoints); }

    // device
    {
        CUDA_CHECK(cudaFree(d_res));

        CUDA_CHECK(cudaFree(d_nVoxelVertsArray));
        CUDA_CHECK(cudaFree(d_nVoxelVertsScan);)

        CUDA_CHECK(cudaFree(d_isValidVoxelArray));
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

void MC::launch_determineVoxelKernel(const uint &nVoxels,
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
                    d_nVoxelVertsArray, d_voxelCubeIndex, d_voxelSDF, d_isValidVoxelArray);
    getLastCudaError("Kernel: 'determineVoxelKernel' failed!\n");
#ifndef NDEBUG
    cudaDeviceSynchronize();
#endif

    d_thrustExclusiveScan(nVoxels, d_nVoxelVertsArray, d_nVoxelVertsScan);
    d_thrustExclusiveScan(nVoxels, d_isValidVoxelArray, d_nValidVoxelsScan);

    uint lastElement, lastScanElement;
    CUDA_CHECK(cudaMemcpy(&lastElement, d_isValidVoxelArray + nVoxels - 1,
                          sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nValidVoxelsScan + nVoxels - 1,
                          sizeof(uint), cudaMemcpyDeviceToHost));
    nValidVoxels = lastElement + lastScanElement;
    if (nValidVoxels == 0)
        return;

    CUDA_CHECK(cudaMemcpy(&lastElement, d_nVoxelVertsArray + nVoxels - 1,
                          sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nVoxelVertsScan + nVoxels - 1,
                          sizeof(uint), cudaMemcpyDeviceToHost));
    allTriVertices = lastElement + lastScanElement;
}

void MC::launch_compactVoxelsKernel(const int &nVoxels) {
    CUDA_CHECK(
            cudaMalloc((void **) &d_compactedVoxelArray, sizeof(uint) * nVoxels));

    dim3 nThreads(NTHREADS, 1, 1);
    dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
    while (nBlocks.x > 65535) {
        nBlocks.x /= 2;
        nBlocks.y *= 2;
    }

    MCKernel::compactVoxels<<<nBlocks, nThreads>>>(
            nVoxels, d_isValidVoxelArray, d_nValidVoxelsScan, d_compactedVoxelArray);
    getLastCudaError("Kernel: 'compactVoxelsKernel' failed!\n");
}

void MC::launch_voxelToMeshKernel(const uint &maxVerts,
                                  const uint &nVoxels) {
    if(nValidVoxels == 0) return;

    dim3 nThreads(NTHREADS, 1, 1);
    dim3 nBlocks((nValidVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
    while (nBlocks.x > 65535) {
        nBlocks.x /= 2;
        nBlocks.y *= 2;
    }

    MCKernel::voxelToMeshKernel<<<nBlocks, nThreads>>>(
            nValidVoxels, maxVerts, d_isoVal, d_voxelSize, d_gridOrigin, d_res,
                    d_compactedVoxelArray, nVertsTex, triTex, d_voxelCubeIndex, d_voxelSDF,
                    d_nVoxelVertsScan, d_triPoints);
    getLastCudaError("Kernel: 'voxelToMeshKernel' failed!\n");
    CUDA_CHECK(cudaMemcpy(h_triPoints, d_triPoints, sizeof(double3) * maxVerts,
                          cudaMemcpyDeviceToHost));
}

void MC::writeToOBJFile(const std::string &filename) {
    checkDir(filename);
    std::ofstream out(filename);
    if (!out) {
        fprintf(stderr, "IO Error: File %s could not be opened!\n",
                filename.c_str());
        return;
    }

    printf("The number of mesh's vertices = %d\n", allTriVertices);
    printf("The number of mesh's faces = %d\n", allTriVertices / 3);
    for (int i = 0; i < allTriVertices; i += 3) {
        const int faceIdx = i;

        out << "v " << h_triPoints[i].x << ' ' << h_triPoints[i].y << ' '
            << h_triPoints[i].z << '\n';
        out << "v " << h_triPoints[i + 1].x << ' ' << h_triPoints[i + 1].y << ' '
            << h_triPoints[i + 1].z << '\n';
        out << "v " << h_triPoints[i + 2].x << ' ' << h_triPoints[i + 2].y << ' '
            << h_triPoints[i + 2].z << '\n';

        out << "f " << faceIdx + 1 << ' ' << faceIdx + 2 << ' ' << faceIdx + 3
            << '\n';
    }

    out.close();
}

void MC::marching_cubes(const uint3 &resolution, const double3 &gridOrigin,
                        const double3 &voxelSize, const double &isoVal,
                        const bool &sdfFlag, const std::string &filename,
                        const std::vector<double> &h_voxelSDF) {
    uint nVoxels = resolution.x * resolution.y * resolution.z;
    uint maxVerts = nVoxels * 18;

    using namespace std::chrono;
    time_point<system_clock> start, end;

    start = system_clock::now();

    initResources(sdfFlag, resolution, nVoxels, isoVal, gridOrigin, voxelSize,
                  maxVerts, h_voxelSDF);

    launch_determineVoxelKernel(nVoxels, isoVal, maxVerts);
    if (allTriVertices == 0) {
        printf("There is no valid vertices...\n");
        return;
    }

    launch_compactVoxelsKernel(nVoxels);

    launch_voxelToMeshKernel(maxVerts, nVoxels);

    end = system_clock::now();
    duration<double> elapsed_seconds = end - start;
    std::time_t end_time = system_clock::to_time_t(end);
    std::cout << "Finished computation at " << std::ctime(&end_time)
              << "Elapsed time: " << elapsed_seconds.count() << "s\n----------\n";

    std::cout << "Write to obj..." << std::endl;
    writeToOBJFile(filename);

    freeResources();
}
