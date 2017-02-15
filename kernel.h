#pragma once

#include <stdint.h>

enum cm_type {
    CM_SQR_GENERIC,
    CM_CUBE_GENERIC,
    CM_FULL_GENERIC,
    CM_BURNING_SHIP_GENERIC,

    CM_SQR_FLOAT, // these are for timing/testing purposes
    CM_SQR_DOUBLE,
    CM_FRACTAL_TYPES_END,
};

enum cm_colors {
    CM_DIST_SNOWFLAKE,
    CM_DIST_GREEN_BLUE,
    CM_DIST_BLACK_BROWN_BLUE,
    CM_COLOR_DIST_END,

    CM_ITER_BLACK_BROWN_BLUE,

    CM_COLORS_END
};

// NOTE: do not use kernel_params for GPU code since it increases register pressure
struct kernel_params {
    cm_type type;
    cm_colors color;

    uint32_t* image_buffer;
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t bufferWidth; // accounts for multiple samples per pixel
    uint32_t bufferHeight;
    uint32_t sqrtSamples;

    uint32_t iter;
    double centerX;
    double centerY;
    double zoom;
    double bailout;
    double z0_x;
    double z0_y;
    double exponent;

    kernel_params() {
        type = CM_SQR_GENERIC;
        color = CM_ITER_BLACK_BROWN_BLUE;
        image_buffer = nullptr;
        imageWidth = 1920;
        imageHeight = 1080;
        sqrtSamples = 2;
        bufferWidth  = sqrtSamples * imageWidth;
        bufferHeight = sqrtSamples * imageHeight;
        iter = 256;
        centerX = -0.5;
        centerY = 0.0;
        zoom = 1.0;
        z0_x = 0.0;
        z0_y = 0.0;
        bailout = 64.0;
        exponent = 2.0;
    }
};

extern "C" {
    void LaunchFractalKernel(const kernel_params &p);
    void LaunchResamplingKernel(uint32_t * src, uint32_t * dest, uint32_t target_width, uint32_t target_height, uint32_t n);
    float GenerateKernelLanczosWeights(uint32_t n);
}
