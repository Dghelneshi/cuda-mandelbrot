#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdint.h>
#include <stdio.h>

#include "kernel.h"
#include "float.h"
#include "mandelbrot.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8

__device__ uint32_t FloatToRGBA(float r, float g, float b) {
    r = clamp(r, 0.0f, 1.0f) * 255.99f;
    g = clamp(g, 0.0f, 1.0f) * 255.99f;
    b = clamp(b, 0.0f, 1.0f) * 255.99f;
    return (0xFFu << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b); // ARGB in register -> BGRA in memory
}
__device__ uint32_t FloatToRGBA(float3 &c) {
    return FloatToRGBA(c.x, c.y, c.z);
}

__device__ float3 RGBAtoFloat(uint32_t c) {
    float div = 1.0f / 255.0f;
    float r = ((c >> 16) & 0xFF) * div;
    float g = ((c >>  8) & 0xFF) * div;
    float b = ((c >>  0) & 0xFF) * div;
    return make_float3(r, g, b);
}

#define M_PI_F 3.14159265f
static __host__ __device__ float sinc(float x) {
    return sinpif(x) / (M_PI_F * x);
}
// Lanczos sampling weights
static __host__ __device__ float Lanczos(float x, float y, float n) {
    float d = sqrtf(x*x + y*y);
    if (d == 0.0f)
        return 1.0f;
    else if (fabs(d) >= n)
        return 0.0f;
    else
        return sinc(d) * sinc(d / n);
}

#if LANCZOS_CPU_PRECALC
#define MAX_SAMPLES 1024
static __constant__ float lanczosWeights_const[MAX_SAMPLES];
#endif

// Downscales an image by n in each dimension
// Note: This is a somewhat naive implementation of Lanczos, using an ever increasing amount of lobes (samples).
// TODO: Having a fixed lobe count and applying the resampling multiple times should decrease ringing artifacts.
//       Could also start threads depending on source image size and do a reduction-type kernel. Should be faster.
__global__ void LanczosResample(uint32_t * __restrict__ src, uint32_t * __restrict__ dest,
                                uint32_t target_width, uint32_t target_height, uint32_t n) {
    // target x and y coordinates
    uint32_t tx = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t ty = blockIdx.y*blockDim.y + threadIdx.y;

    if (tx >= target_width || ty >= target_height) return; // image does not necessarily fit nicely into blocks

    uint32_t src_width = target_width * n;
    float fn = float(n);
    float totalLanczosWeight = 0.0f;

    // stretch the kernel slightly depending on sample count
    // improves image sharpness at medium to high spp, but might introduce more ringing.
    float stretch = sqrtf(fn) * 0.5f;
    
    float3 c = make_float3(0.0f);
    for (uint32_t sampleY = 0; sampleY < n; ++sampleY) {
        for (uint32_t sampleX = 0; sampleX < n; ++sampleX) {
            uint32_t sourceX = tx * n + sampleX;
            uint32_t sourceY = ty * n + sampleY;

#ifdef LANCZOS_CPU_PRECALC
            float weight = lanczosWeights_const[sampleY * n + sampleX];
#else
            // x and y offsets of samples, used to get lanczos weight
            float u = ((sampleY + 0.5f) / fn - 0.5f);
            float v = ((sampleX + 0.5f) / fn - 0.5f);
            float weight = Lanczos(u * stretch, v * stretch, fn);
#endif
            c += weight * RGBAtoFloat(src[sourceY * src_width + sourceX]);
            totalLanczosWeight += weight;
        }
    }
    c = c / totalLanczosWeight;
    dest[ty * target_width + tx] = FloatToRGBA(c);
}

// kernel using templates to avoid the switch in GPU code, uses distance function for color
template<typename T, cm_type V, cm_colors C>
__global__ void FractalKernel(uint32_t *image_buffer, uint32_t w, uint32_t h, T centerX, T centerY, T zoom,
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
        T dist = MandelbrotDist<T, V>()(x, y, bailout, z0_x, z0_y, iter, exponent);
        f = float(dist / zoom);
    }
    else {
        f = MandelbrotSIter<T, V>()(x, y, bailout, z0_x, z0_y, iter, exponent);
        // normalize iteration count, this prevents noisy images
        f = f * (256.0f / float(iter));
    }
    rgb = ColorizeMandelbrot<C>(f);
    image_buffer[iy * w + ix] = FloatToRGBA(rgb);
}


// these functions handle the host side switching
template<typename T, cm_type V>
void LaunchFractalKernelTemplate(const kernel_params &p, const dim3 &block, const dim3 &grid) {

    // specialize our kernel template one step at a time,
    // third step: choose the fractal coloring function C and launch the kernel!
    switch (p.color) {
    case CM_DIST_BLACK_BROWN_BLUE:
        FractalKernel<T, V, CM_DIST_BLACK_BROWN_BLUE><<<grid,block>>>(p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
        break;
    case CM_DIST_GREEN_BLUE:
        FractalKernel<T, V, CM_DIST_GREEN_BLUE> << <grid, block >> >(p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
        break;
    case CM_DIST_SNOWFLAKE:
        FractalKernel<T, V, CM_DIST_SNOWFLAKE> << <grid, block >> >(p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
        break;
    case CM_ITER_BLACK_BROWN_BLUE:
        FractalKernel<T, V, CM_ITER_BLACK_BROWN_BLUE> << <grid, block >> >(p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
        break;
    }
}

template<typename T>
void LaunchFractalKernelTemplate(const kernel_params &p, const dim3 &block, const dim3 &grid) {
    
    // specialize our kernel template one step at a time,
    // second step: choose the fractal type V
    switch (p.type) {
    case CM_SQR_GENERIC:
        LaunchFractalKernelTemplate<T, CM_SQR_GENERIC>(p, block, grid);
        break;
    case CM_CUBE_GENERIC:
        LaunchFractalKernelTemplate<T, CM_CUBE_GENERIC>(p, block, grid);
        break;
    case CM_FULL_GENERIC:
        LaunchFractalKernelTemplate<T, CM_FULL_GENERIC>(p, block, grid);
        break;
    case CM_BURNING_SHIP_GENERIC:
        LaunchFractalKernelTemplate<T, CM_BURNING_SHIP_GENERIC>(p, block, grid);
        break;

    // test kernels
    case CM_SQR_FLOAT:
        LaunchFractalKernelTemplate<float, CM_SQR_FLOAT>(p, block, grid);
        break;
    case CM_SQR_DOUBLE:
        LaunchFractalKernelTemplate<double, CM_SQR_DOUBLE>(p, block, grid);
        break;
    }
}

extern "C" {

    void LaunchFractalKernel(const kernel_params &p) {
        
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((p.bufferWidth + block.x - 1) / block.x, (p.bufferHeight + block.y - 1) / block.y, 1);
        
        // choose double or float arithmetic according to the mandelbrot coordinate difference between pixels
        double pixelSize = 2.0 * (p.zoom / double(p.bufferHeight));
        double threshold = FLT_EPSILON * 4.0; // computing the threshold exactly is a pain, this is a fine approximation

#if !defined(CM_TEST_SWITCH)
        // specialize our kernel template one step at a time
        // first step: choose the data type T
        if (pixelSize > threshold) {
            LaunchFractalKernelTemplate<float>(p, block, grid);
        }
        else {
            LaunchFractalKernelTemplate<double>(p, block, grid);
        }
#else
        // performance comparison between full templates and switch in kernel
        if (pixelSize > threshold)
            SwitchKernel<float> << <grid, block >> >(p.type, p.color, p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
        else
            SwitchKernel<double> << <grid, block >> >(p.type, p.color, p.image_buffer, p.bufferWidth, p.bufferHeight, T(p.centerX), T(p.centerY), T(p.zoom), T(p.bailout), T(p.z0_x), T(p.z0_y), p.iter, T(p.exponent));
#endif
    }


    // generates the sample distribution for image downscaling, including Lanczos weights
    // should be called at some point before launching the kernel if LANCZOS_CPU_PRECALC is defined
    // doesn't seem to be faster than the naive version, memory latency is much more limiting than compute
    float GenerateKernelLanczosWeights(uint32_t n) {
#ifdef LANCZOS_CPU_PRECALC
        float *lanczosWeights = (float*) malloc(n*n*sizeof(*lanczosWeights));
        float fn = float(n);
        float totalLanczosWeight = 0.0f;

        // stretch the kernel slightly depending on sample count. this improves image sharpness at medium to high spp.
        // do not touch without testing extensively, can have disastrous results!
        // sqrtf(n) * 0.5f improves quality at around 25 - 64 spp, with more samples image becomes too sharp
        float stretch = sqrtf(fn) * 0.5f;

        for (uint32_t y = 0; y < n; y++) {
            for (uint32_t x = 0; x < n; x++) {
                // sample distribution is a regular grid
                float u_adjust = ((x + 0.5f) / fn - 0.5f);
                float v_adjust = ((y + 0.5f) / fn - 0.5f);
                float weight = lanczos(u_adjust * stretch, v_adjust * stretch, fn);
                lanczosWeights[y * n + x] = weight;

                totalLanczosWeight += weight;
            }
        }

        cudaMemcpyToSymbol(lanczosWeights_const, lanczosWeights, n*n*sizeof(*lanczosWeights));

        free(lanczosWeights);
        return totalLanczosWeight;
#else
        return 0.0f;
#endif
    }


    void LaunchResamplingKernel(uint32_t * src, uint32_t * dest, uint32_t target_width, uint32_t target_height, uint32_t n) {

        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dim3 grid((target_width + block.x - 1) / block.x, (target_height + block.y - 1) / block.y, 1);

        LanczosResample << < grid, block >> > (src, dest, target_width, target_height, n);

    }
}