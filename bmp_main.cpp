#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "kernel.h"
#include "bmp_output.h"
#include "timing.h"
#include <windows.h>

#define PERF_PRINT_FORMAT "%26s%8.3fms\n"

int bmpMain(const char* filename, kernel_params &params) {

    uint64_t t1, tStart = GetTime();

    if (!filename) filename = "output.bmp";
    
    params.bufferWidth  = params.sqrtSamples * params.imageWidth;
    params.bufferHeight = params.sqrtSamples * params.imageHeight;
    size_t bufferSize = size_t(params.bufferWidth) * size_t(params.bufferHeight) * sizeof(*params.image_buffer);
    size_t resultSize = size_t(params.imageWidth)  * size_t(params.imageHeight)  * sizeof(*params.image_buffer);

    uint32_t* cpuBuffer;    // image in CPU memory
    uint32_t* gpuBuffer;    // image in GPU memory
    uint32_t* resultBuffer; // result after downsampling (if sqrtSamples > 1)
    cpuBuffer = (uint32_t*) malloc(resultSize);
    cudaMalloc(&gpuBuffer, bufferSize);
    params.image_buffer = gpuBuffer;

    printf(PERF_PRINT_FORMAT, "Buffer allocation: ", TimeDelta(tStart, GetTime()));
        
    cuda_timer t = StartCudaTimer();
    LaunchFractalKernel(params);
    printf(PERF_PRINT_FORMAT, "Fractal kernel: ", StopCudaTimer(t));
    
    if (params.sqrtSamples > 1) {

        t1 = GetTime();
        cudaMalloc(&resultBuffer, resultSize);
        printf(PERF_PRINT_FORMAT, "Result buffer allocation: ", TimeDelta(t1, GetTime()));
        t = StartCudaTimer();

        GenerateKernelLanczosWeights(params.sqrtSamples);
        LaunchResamplingKernel(gpuBuffer, resultBuffer, params.imageWidth, params.imageHeight, params.sqrtSamples);

        printf(PERF_PRINT_FORMAT, "Resampling kernel: ", StopCudaTimer(t));

        cudaFree(gpuBuffer);
    }
    else {
        resultBuffer = gpuBuffer;
    }

    t1 = GetTime();
    cudaMemcpy(cpuBuffer, resultBuffer, resultSize, cudaMemcpyDeviceToHost);
    printf(PERF_PRINT_FORMAT, "Memcpy to host: ", TimeDelta(t1, GetTime()));

    t1 = GetTime();
    int res = write_bmp(filename, params.imageWidth, params.imageHeight, (uint8_t*) cpuBuffer);
    
    if (res != 0)
        printf("Error: Could not open file %s\n", filename);

    printf(PERF_PRINT_FORMAT, "Bitmap write: ", TimeDelta(t1, GetTime()));

    free(cpuBuffer);
    cudaFree(resultBuffer);

    printf(PERF_PRINT_FORMAT, "Total time spent: ", TimeDelta(tStart, GetTime()));

    return 0;
}
