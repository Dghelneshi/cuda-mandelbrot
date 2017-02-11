#include <cuda_runtime.h>
#include "timing.h"

#ifdef _WIN32

    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>

    static LARGE_INTEGER timerFreq;

    void initTime() {
        QueryPerformanceFrequency(&timerFreq);
    }

    uint64_t getTime() {
        LARGE_INTEGER t1;
        QueryPerformanceCounter(&t1);
        return t1.QuadPart;
    }

    // returns milliseconds
    float timeDelta(uint64_t start, uint64_t stop) {
        return float(((stop - start) * 1000) / (double) timerFreq.QuadPart);
    }

#else

    #include <time.h>

    void initTime() {}

    uint64_t getTime() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return (uint64_t) ts.tv_nsec + (uint64_t) ts.tv_sec * 1000000000ull;
    }

    // returns milliseconds
    float timeDelta(uint64_t start, uint64_t stop) {
        return ((stop - start) / 1000000.0);
    }

#endif

cudaTimer startCudaTimer() {
    cudaTimer t;
    cudaEventCreate(&t.start);
    cudaEventCreate(&t.stop, cudaEventBlockingSync);
    cudaEventRecord(t.start);
    return t;
}

// blocks until gpu is finished, returns milliseconds
float stopCudaTimer(cudaTimer t) {
    float result = 0.0f;
    cudaEventRecord(t.stop);
    cudaEventSynchronize(t.stop);
    cudaEventElapsedTime(&result, t.start, t.stop);
    cudaEventDestroy(t.start);
    cudaEventDestroy(t.stop);
    return result;
}