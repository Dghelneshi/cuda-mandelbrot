#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "kernel.h"
#include "bmp_output.h"
#include "timing.h"


int bmpMain(const char* filename, kernel_params &params) {

    if (!filename) filename = "output.bmp";

    // TODO: implement proper supersampling like in GL window
    params.width  = params.sqrtSamples * params.width;
    params.height = params.sqrtSamples * params.height;

    size_t bufSize = size_t(params.width) * size_t(params.height) * sizeof(*params.image_buffer);

    uint32_t* cpuBuffer; // image in CPU memory
    uint32_t* gpuBuffer; // image in GPU memory
    cpuBuffer = (uint32_t*) malloc(bufSize);
    cudaMalloc((void**) &gpuBuffer, bufSize);

    params.image_buffer = gpuBuffer;
    
    cudaProfilerStart();
    cudaTimer t = startCudaTimer();

    LaunchKernel(params);

    float time = stopCudaTimer(t);
    printf("Generated image in %fms GPU time\n", time);

    cudaMemcpy(cpuBuffer, gpuBuffer, bufSize, cudaMemcpyDeviceToHost);
    int res = write_bmp(filename, params.width, params.height, (uint8_t*) cpuBuffer);
    if (res != 0)
        printf("Error: Could not open file %s\n", filename);

    free(cpuBuffer);
    cudaFree(gpuBuffer);

    return 0;
}
