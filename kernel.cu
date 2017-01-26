
#ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #define snprintf sprintf_s
#else
    #define VK_ESCAPE 27
#endif


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <device_functions.h>
#include <math_functions.h>

#include "helper_math.h"

#include <inttypes.h>

#include "double2_inline.h"
#include "kernel.h"

static const uint32_t blockSizeX = 16;
static const uint32_t blockSizeY = 8;


__device__ uint32_t rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 1.0f) * 255.99f;
    g = clamp(g, 0.0f, 1.0f) * 255.99f;
    b = clamp(b, 0.0f, 1.0f) * 255.99f;
    return (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b); // ARGB in register -> BGRA in memory
}
__device__ uint32_t rgbToInt(float3 &c)
{
    return rgbToInt(c.x, c.y, c.z);
}


__device__ double mandelbrot(double x, double y, double zoom, double maxlen2) {
    
    double2 c = make_double2(x, y);
    double2 z = make_double2(0.0, 0.0);
    double2 dz = make_double2(0.0, 0.0); // derivative z'
    double len2 = 0.0;
    
    for (int i = 0; i < 256; i++) {

        // z' = 2*z*z' + 1
        dz = 2.0 * make_double2(z.x*dz.x - z.y*dz.y, z.x*dz.y + z.y*dz.x) + make_double2(1.0, 0.0);

        // z = z^2 + c
        z = make_double2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;

        len2 = dot(z, z);

        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    double d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);
    
    return d;
}

__device__ double mandelbrot3(double x, double y, double zoom, double maxlen2) {

    double2 c = make_double2(x, y);
    double2 z = make_double2(0.0, 0.0);
    double2 dz = make_double2(0.0, 0.0); // derivative z'
    double2 z2;
    double len2 = 0.0;

    for (int i = 0; i < 256; i++) {

        // z' = 3*z^2*z' + 1
        z2 = make_double2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        dz = 3.0 * make_double2(z2.x * dz.x - z2.y * dz.y, z2.x * dz.y + z2.y * dz.x) + make_double2(1.0, 0.0);

        // z = z^3 + c
        // z^3 = x^3 - 3 x y^2 + i (3 x^2 y - y^3)
        z = make_double2(z.x*z.x*z.x - 3.0 * z.x * z.y*z.y, 3.0 * z.x*z.x*z.y - z.y*z.y*z.y) + c;

        len2 = dot(z, z);

        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    double d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);

    return d;
}


__global__ void kernel(uint32_t *image_buffer, uint32_t w, uint32_t h, double centerX, double centerY, double zoom, double maxlen2) {
    // image x and y coordinates
    uint32_t ix = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y*blockDim.y + threadIdx.y;

    if (ix >= w || iy >= h) return; // image does not necessarily fit nicely into blocks

    double hw = w * 0.5;
    double hh = h * 0.5;
    // normalized image coordinates
    // divide both by height for correct aspect ratio
    double nx = (ix - hw) / hh;
    double ny = (iy - hh) / hh;
    // function x and y coordinates
    double x = zoom * nx + centerX;
    double y = zoom * ny + centerY;

    double dist = mandelbrot(x, y, zoom, maxlen2);

    // do some soft coloring based on distance
    dist = clamp(12.0 * dist / zoom, 0.0, 1.0);
    dist = rcbrt(rsqrt(dist)); // dist^(1/6)

    float3 rgb = make_float3(0, cos(dist * 3.14159), sin(dist * 3.14159));

    image_buffer[iy * w + ix] = rgbToInt(rgb);
}

extern "C" {

    void launchKernel(uint32_t* image_buffer, uint32_t w, uint32_t h, pos_t pos, double maxlen2) {
        dim3 block(blockSizeX, blockSizeY, 1);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

        kernel << <grid, block >> >(image_buffer, w, h, pos.centerX, pos.centerY, pos.zoom, maxlen2);
    }

}