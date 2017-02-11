#include "kernel.h"
#include "bmp_output.h"
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "timing.h"
#include <stdio.h>

#define WIDTH 1920
#define HEIGHT 1080
#define BUFSIZE (WIDTH * HEIGHT * BYTESPERPIXEL)

int bmpMain() {
    uint32_t* cpuBuffer; // image in CPU memory
    uint32_t* gpuBuffer; // image in GPU memory
    cpuBuffer = (uint32_t*) malloc(BUFSIZE);
    cudaMalloc((void**)&gpuBuffer, BUFSIZE);

    kernel_params params;
    params.image_buffer = gpuBuffer;
    params.width  = WIDTH;
    params.height = HEIGHT;

    cudaTimer t = startCudaTimer();

    LaunchKernel(params);

    float time = stopCudaTimer(t);
    printf("Generated image in %fms\n", time);

    cudaMemcpy(cpuBuffer, gpuBuffer, BUFSIZE, cudaMemcpyDeviceToHost);
    write_bmp("output.bmp", WIDTH, HEIGHT, (uint8_t*) cpuBuffer);

    free(cpuBuffer);
    cudaFree(gpuBuffer);

    return 0;
}
