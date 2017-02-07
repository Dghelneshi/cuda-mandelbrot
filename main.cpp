#ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #define snprintf sprintf_s
#endif


// includes, OpenGL
#ifndef CM_NOGL
#include <GL/glut.h>
#include "window_gl.h"
#endif

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "helper_cuda.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#include "timing.h"
#include "main.h"
#include "kernel.h"




void terminate(int code) {
    cudaProfilerStop();
    cudaDeviceReset();
    exit(code);
}

int main(int argc, const char* argv[]) {

    cudaProfilerStop();
    initTime();

    //if (argc > 1 && (strcmp(argv[1], "-gl") == 0)) {
    if (true) {
#ifndef CM_NOGL
        if (!initGLWindow(argc, argv))
            return 1;

        cudaProfilerStart();
        // start rendering main-loop
        glutMainLoop();
#endif
    }
    else {
#ifdef CM_NOGL
				cudaProfilerStart();	
				//do the BMP stuff
				bmpMain(argc, argv);
#endif
    }

    terminate(0);

    return 0;
}
