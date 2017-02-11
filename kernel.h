#pragma once

#include <stdint.h>

// NOTE: do not use kernel_params for GPU code since it increases register pressure
struct kernel_params {
    uint32_t* image_buffer;
    uint32_t width;
    uint32_t height;
    int iter;
    double centerX;
    double centerY;
    double zoom;
    double maxlen2;
    double startX;
    double startY;
    double exponent;

    kernel_params() {
        image_buffer = nullptr;
        width = 1920;
        height = 1080;
        centerX = -0.5;
        centerY = 0.0;
        zoom = 1.0;
        startX = 0.0;
        startY = 0.0;
        maxlen2 = 64.0;
        exponent = 2.0;
        iter = 256;
    }
};

extern "C" {
    void LaunchKernel(kernel_params p);
}
