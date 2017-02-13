#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "timing.h"
#include "bmp_main.h"
#include "kernel.h"
#include "cmdline_parser.h"

#ifndef CM_NOGL
#include "window_gl.h"
#endif


int main(int argc, char* argv[]) {

    cudaProfilerStop();
    initTime();

    kernel_params p;
    bool useGL = false;
    const char* outFile = nullptr;

    parseArgs(argc, argv, &p, &useGL, outFile);
    
    if (useGL) {
        #ifdef CM_NOGL
            printf("Error: This executable was compiled without OpenGL support.");
        #else
            GLWindowMain(argc, argv, p);
        #endif
    }
    else {
        bmpMain(outFile, p);
    }

    cudaProfilerStop();
    cudaDeviceReset();

    return 0;
}
