#pragma once

#include <stdint.h>
#include <cuda_profiler_api.h>

struct cudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

void initTime();
uint64_t getTime();
float timeDelta(uint64_t start, uint64_t stop);

cudaTimer startCudaTimer();
float stopCudaTimer(cudaTimer t); // blocks until gpu is finished