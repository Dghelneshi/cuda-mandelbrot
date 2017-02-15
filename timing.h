#pragma once

#include <stdint.h>
#include <cuda_profiler_api.h>

struct cuda_timer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

void InitTime();
uint64_t GetTime();
float TimeDelta(uint64_t start, uint64_t stop); // returns milliseconds

cuda_timer StartCudaTimer();
float StopCudaTimer(cuda_timer t); // blocks until gpu is finished, returns milliseconds