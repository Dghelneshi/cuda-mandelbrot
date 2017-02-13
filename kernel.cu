#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "kernel.h"
#include "float.h"
#include "mandelbrot.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

__device__ uint32_t RgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 1.0f) * 255.99f;
    g = clamp(g, 0.0f, 1.0f) * 255.99f;
    b = clamp(b, 0.0f, 1.0f) * 255.99f;
    return (0xFFu << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b); // ARGB in register -> BGRA in memory
}
__device__ uint32_t RgbToInt(float3 &c)
{
    return RgbToInt(c.x, c.y, c.z);
}

// kernel using a switch for the different mandelbrot types. used as comparison to template solution.
template<typename T>
__global__ void SwitchKernel(cm_type t, cm_colors c, uint32_t *image_buffer, uint32_t w, uint32_t h, T centerX, T centerY, T zoom,
                             T bailout, T z0_x, T z0_y, int iter, T exponent) {

    // image x and y coordinates
    uint32_t ix = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y*blockDim.y + threadIdx.y;

    if (ix >= w || iy >= h) return; // image does not necessarily fit nicely into blocks

    T hw = w * (T)0.5;
    T hh = h * (T)0.5;
    // normalized image coordinates, y goes from -1 to 1, x is scaled by aspect
    T nx = (ix - hw) / hh;
    T ny = (iy - hh) / hh;
    // function x and y coordinates
    T x = zoom * nx + centerX;
    T y = zoom * ny + centerY;

    T dist;
    switch (t) {
    case CM_SQR_GENERIC:
        dist = MandelbrotDistSquareGeneric<T>(x, y, bailout, z0_x, z0_y, iter);
        break;
    case CM_CUBE_GENERIC:
        dist = MandelbrotDistCubeGeneric<T>(x, y, bailout, z0_x, z0_y, iter);
        break;
    case CM_FULL_GENERIC:
        dist = MandelbrotDistFullGeneric<T>(x, y, bailout, z0_x, z0_y, iter, exponent);
        break;
    case CM_SQR_FLOAT:
        dist = MandelbrotDistSquareFloat(x, y, bailout, z0_x, z0_y, iter);
        break;
    case CM_SQR_DOUBLE:
        dist = MandelbrotDistSquareDouble(x, y, bailout, z0_x, z0_y, iter);
        break;
    }

    float3 rgb;
    switch (c) {
    case CM_DIST_BLACK_BROWN_BLUE:
        rgb = ColorizeMandelbrot<CM_DIST_BLACK_BROWN_BLUE>(float(dist / zoom));
        break;
    case CM_DIST_GREEN_BLUE:
        rgb = ColorizeMandelbrot<CM_DIST_GREEN_BLUE>(float(dist / zoom));
        break;
    case CM_DIST_SNOWFLAKE:
        rgb = ColorizeMandelbrot<CM_DIST_SNOWFLAKE>(float(dist / zoom));
        break;
    }

    image_buffer[iy * w + ix] = RgbToInt(rgb);
}

// kernel using templates to avoid the switch in GPU code, uses distance function for color
template<typename T, cm_type M, cm_colors C>
__global__ void TemplateKernel(uint32_t *image_buffer, uint32_t w, uint32_t h, T centerX, T centerY, T zoom,
                                   T bailout, T z0_x, T z0_y, int iter, T exponent) {

    // image x and y coordinates
    uint32_t ix = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t iy = blockIdx.y*blockDim.y + threadIdx.y;

    if (ix >= w || iy >= h) return; // image does not necessarily fit nicely into blocks

    T hw = w * T(0.5);
    T hh = h * T(0.5);
    // normalized image coordinates, y goes from -1 to 1, x is scaled by aspect
    T nx = (ix - hw) / hh;
    T ny = (iy - hh) / hh;
    // function x and y coordinates
    T x = zoom * nx + centerX;
    T y = zoom * ny + centerY;

    float3 rgb;
    float f;

    // pick iteration function depending on whether we want a distance estimate or smooth iteration count for coloring
    // the compiler will optimize out the branch that is not taken in each template instantiation
    if (C < CM_COLOR_DIST_END) {
        T dist = MandelbrotDist<T, M>(x, y, bailout, z0_x, z0_y, iter, exponent);
        f = float(dist / zoom);
    }
    else {
        f = MandelbrotSIter<T, M>(x, y, bailout, z0_x, z0_y, iter, exponent);
        // normalize iteration count, this prevents noisy images
        f = f * (256.0f / float(iter));
    }
    rgb = ColorizeMandelbrot<C>(f);
    image_buffer[iy * w + ix] = RgbToInt(rgb);
}


// these functions handle the host side switching
template<typename T, cm_type M>
void LaunchTemplateKernel(const kernel_params &p, dim3 &block, dim3 &grid) {
    switch (p.color) {
    case CM_DIST_BLACK_BROWN_BLUE:
        TemplateKernel<T, M, CM_DIST_BLACK_BROWN_BLUE><<<grid,block>>>(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        break;
    case CM_DIST_GREEN_BLUE:
        TemplateKernel<T, M, CM_DIST_GREEN_BLUE> << <grid, block >> >(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        break;
    case CM_DIST_SNOWFLAKE:
        TemplateKernel<T, M, CM_DIST_SNOWFLAKE> << <grid, block >> >(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        break;
    case CM_ITER_BLACK_BROWN_BLUE:
        TemplateKernel<T, M, CM_ITER_BLACK_BROWN_BLUE> << <grid, block >> >(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        break;
    }
}

template<typename T>
void LaunchTemplateKernel(const kernel_params &p, dim3 &block, dim3 &grid) {
    switch (p.type) {
    case CM_SQR_GENERIC:
        LaunchTemplateKernel<T, CM_SQR_GENERIC>(p, block, grid);
        break;
    case CM_CUBE_GENERIC:
        LaunchTemplateKernel<T, CM_CUBE_GENERIC>(p, block, grid);
        break;
    case CM_FULL_GENERIC:
        LaunchTemplateKernel<T, CM_FULL_GENERIC>(p, block, grid);
        break;
    case CM_BURNING_SHIP_GENERIC:
        LaunchTemplateKernel<T, CM_BURNING_SHIP_GENERIC>(p, block, grid);
        break;

        // test kernels
    case CM_SQR_FLOAT:
        LaunchTemplateKernel<float, CM_SQR_FLOAT>(p, block, grid);
        break;
    case CM_SQR_DOUBLE:
        LaunchTemplateKernel<double, CM_SQR_DOUBLE>(p, block, grid);
        break;
    }
}

extern "C" {

    void LaunchKernel(const kernel_params &p) {
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((p.width + block.x - 1) / block.x, (p.height + block.y - 1) / block.y, 1);
        
        // choose double or float arithmetic according to the mandelbrot coordinate difference between pixels
        double pixelSize = 2.0 * (p.zoom / p.height);
        double threshold = FLT_EPSILON * 4.0; // computing the threshold exactly is a pain, this is a fine approximation

#ifdef CM_TEST_SWITCH
        if (pixelSize > threshold)
            SwitchKernel<float><<<grid,block>>>(p.type, p.color, p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        else
            SwitchKernel<double><<<grid,block>>>(p.type, p.color, p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.bailout, p.z0_x, p.z0_y, p.iter, p.exponent);
        return;
#endif

        if (pixelSize > threshold) {
            LaunchTemplateKernel<float>(p, block, grid);
        }
        else {
            LaunchTemplateKernel<double>(p, block, grid);
        }
    }
}