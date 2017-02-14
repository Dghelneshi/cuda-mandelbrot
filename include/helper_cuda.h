/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#pragma once

#include <stdint.h>

#define max(a,b) (((a) > (b)) ? (a) : (b))


inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
        { -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    return nGpuArchCoresPerSM[index - 1].Cores;
}


// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int max_perf_device = 0;
    int device_count = 0;
    int best_arch = 0;
    uint64_t max_compute_perf = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);


    // Find the best major SM Architecture GPU device
    for (int i = 0; i < device_count; i++) {
        cudaGetDeviceProperties(&deviceProp, i);

        if (deviceProp.computeMode != cudaComputeModeProhibited) {
            if (deviceProp.major > 0 && deviceProp.major < 9999) {
                best_arch = max(best_arch, deviceProp.major);
            }
        }
    }

    // find highest perf
    for (int i = 0; i < device_count; i++) {

        cudaGetDeviceProperties(&deviceProp, i);

        if ((deviceProp.computeMode != cudaComputeModeProhibited) && (deviceProp.major == best_arch)) {

            int sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

            uint64_t compute_perf = (uint64_t) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf > max_compute_perf) {
                    max_compute_perf = compute_perf;
                    max_perf_device = i;
            }
        }
    }

    return max_perf_device;
}

#endif
