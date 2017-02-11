#ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #define snprintf sprintf_s
#endif

// includes, OpenGL
#ifndef CM_NOGL
#include <GL/freeglut.h>
#include "window_gl.h"
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "timing.h"
#include "bmp_main.h"


void Exit(int code) {
    cudaProfilerStop();
    cudaDeviceReset();
    exit(code);
}

int main(int argc, const char* argv[]) {

    cudaProfilerStop();
    initTime();

    if (argc > 1 && (strcmp(argv[1], "-gl") == 0)) {
        #ifdef CM_NOGL
            printf("Error: This executable was compiled without OpenGL support.");
        #else
            if (!InitGLWindow(argc, argv))
                return 1;

            cudaProfilerStart();
            // start rendering main-loop
            glutMainLoop();
        #endif
    }
    else {
        cudaProfilerStart();
        bmpMain();
    }

    Exit(0);

    return 0;
}
