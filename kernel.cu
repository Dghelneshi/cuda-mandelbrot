
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
#include <math_functions.h>

#include "kernel.h"
#include "float.h"
#include "mandelbrot_variants.cuh"

#define M_PI_F 3.14159265358979323846f
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

template<typename T>
__global__ void SwitchKernel(cm_variants v, uint32_t *image_buffer, uint32_t w, uint32_t h, T centerX, T centerY, T zoom,
                             T maxlen2, T startX, T startY, int iter, T exponent) {

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
    switch (v) {
    case SQR_GENERIC:
        dist = MandelbrotSquareGeneric<T>(x, y, maxlen2, startX, startY, iter);
        break;
    case SQR_FULL_GENERIC:
        dist = MandelbrotFullGeneric<T>(x, y, maxlen2, startX, startY, iter, exponent);
        break;
    case SQR_FLOAT:
        dist = MandelbrotSquareFloat(x, y, maxlen2, startX, startY, iter);
        break;
    case SQR_DOUBLE:
        dist = MandelbrotSquareDouble(x, y, maxlen2, startX, startY, iter);
        break;
    case CUBE_FLOAT:
        dist = MandelbrotCubeFloat(x, y, maxlen2, startX, startY, iter);
        break;
    case CUBE_DOUBLE:
        dist = MandelbrotCubeDouble(x, y, maxlen2, startX, startY, iter);
        break;
    }
    
    // do some soft coloring based on distance
    float normdist = clamp(12.0f * float(dist / zoom), 0.0f, 1.0f);
    normdist = rcbrt(rsqrt(normdist)); // dist^(1/6)

    float3 rgb = make_float3(0.0f, cos(normdist * M_PI_F), sin(normdist * M_PI_F));

    image_buffer[iy * w + ix] = RgbToInt(rgb);
}


template<typename T, cm_variants V>
__global__ void TemplateKernel(uint32_t *image_buffer, uint32_t w, uint32_t h, T centerX, T centerY, T zoom,
                               T maxlen2, T startX, T startY, int iter, T exponent) {

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
    
    T dist = Mandelbrot<T,V>(x, y, maxlen2, startX, startY, iter, exponent);

    // do some soft coloring based on distance
    float normdist = clamp(12.0f * float(dist / zoom), 0.0f, 1.0f);
    normdist = rcbrt(rsqrt(normdist)); // dist^(1/6)

    float3 rgb = make_float3(0.0f, cos(normdist * M_PI_F), sin(normdist * M_PI_F));

    image_buffer[iy * w + ix] = RgbToInt(rgb);
}

extern "C" {

    void LaunchKernel(const kernel_params &p) {
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((p.width + block.x - 1) / block.x, (p.height + block.y - 1) / block.y, 1);
        
        // TODO: no-autoswitch parameter

        // choose double or float arithmetic according to the mandelbrot coordinate difference between pixels
        double pixelSize = 2.0 * (p.zoom / p.height);
        double threshold = FLT_EPSILON * 4.0; // computing the threshold exactly is a pain, this is a fine approximation
        
        if (pixelSize > threshold) {
            TemplateKernel<float, SQR_FULL_GENERIC> << <grid, block >> >(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.maxlen2, p.startX, p.startY, p.iter, p.exponent);
            //SwitchKernel<float> << <grid, block >> >(SQR_GENERIC, p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.maxlen2, p.startX, p.startY, p.iter, p.exponent);
        }
        else {
            TemplateKernel<double, SQR_FULL_GENERIC> << <grid, block >> >(p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.maxlen2, p.startX, p.startY, p.iter, p.exponent);
            //SwitchKernel<double> << <grid, block >> >(SQR_GENERIC, p.image_buffer, p.width, p.height, p.centerX, p.centerY, p.zoom, p.maxlen2, p.startX, p.startY, p.iter, p.exponent);
        }
    }

}