#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "timing.h"
#include "bmp_main.h"
#include "kernel.h"
#include "cmdline_parser.h"
#include "helper_cuda.h"
#include <cstdlib>

#ifndef CM_NOGL
#include "window_gl.h"
#endif


int main(int argc, char* argv[]) {

    cudaProfilerStop();
    InitTime();
    uint64_t tStart = GetTime();

    kernel_params kp;
    program_params pp;
    ParseArgv(argc, argv, &kp, &pp);

    int i = gpuGetMaxGflopsDeviceId();
    cudaSetDevice(i);

    cudaFree(nullptr); // initialize the CUDA context at a known point to avoid performance measurement artifacts
    
    cudaProfilerStart();

    if (pp.useGl) {
        #ifdef CM_NOGL
            printf("Error: This executable was compiled without OpenGL support.");
        #else
            GLWindowMain(argc, argv, kp, pp.window_width, pp.window_height);
        #endif
    }
    else {
        printf("%26s%8.3fms\n", "CUDA Initialization: ", TimeDelta(tStart, GetTime()));
        bmpMain(pp.outputFile, kp);
    }

    cudaProfilerStop();
    cudaDeviceReset();

    return 0;
}
