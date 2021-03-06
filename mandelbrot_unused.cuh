#include <stdint.h>

// this code is for performance comparisons only
// NOTE: changes in this file will not automatically trigger a rebuild in Visual Studio!

// double precision Mandelbrot with exponent = 2
// returns distance estimation
// z = z^2 + c
template<typename T>
struct MandelbrotDist<T, CM_SQR_DOUBLE> {
    __device__ __forceinline__ T operator()(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent) const {

        double2 c = make_double2(x, y);
        double2 z = make_double2(z0_x, z0_y);
        double2 dz = make_double2(0.0, 0.0); // derivative z'
        double len2 = 0.0;

        int i;
        for (i = 0; i < iter; i++) {

            // z' = 2*z*z' + 1
            dz = 2.0 * make_double2(z.x*dz.x - z.y*dz.y, z.x*dz.y + z.y*dz.x) + make_double2(1.0, 0.0);

            // z = z^2 + c
            z = make_double2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;

            len2 = dot(z, z);

            // if z is too far from the origin, assume divergence
            if (len2 > bailout) break;
        }

        // distance	estimation
        // d(c) = |z|*log|z|/|z'|
        double d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);

        return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
    }
};


// single precision Mandelbrot with exponent = 2
// returns distance estimation
// z = z^2 + c
template<typename T>
struct MandelbrotDist<T, CM_SQR_FLOAT> {
    __device__ __forceinline__ T operator()(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent) const {

        float2 c = make_float2(x, y);
        float2 z = make_float2(z0_x, z0_y);
        float2 dz = make_float2(0.0f, 0.0f); // derivative z'
        float len2 = 0.0f;

        int i;
        for (i = 0; i < iter; i++) {

            // z' = 2*z*z' + 1
            dz = 2.0f * make_float2(z.x*dz.x - z.y*dz.y, z.x*dz.y + z.y*dz.x) + make_float2(1.0f, 0.0f);

            // z = z^2 + c
            z = make_float2(z.x*z.x - z.y*z.y, 2.0f*z.x*z.y) + c;

            len2 = dot(z, z);

            // if z is too far from the origin, assume divergence
            if (len2 > bailout) break;
        }

        // distance	estimation
        // d(c) = |z|*log|z|/|z'|
        float d = 0.5f * sqrt(len2 / dot(dz, dz)) * log(len2);

        return (i == iter) ? 0.0f : d; // estimate can be wrong inside blobs, so use iteration count as well
    }
};


// this is duplicate code, but I want to simplify includes.
__device__ uint32_t RgbToInt_(float r, float g, float b)
{
    r = clamp(r, 0.0f, 1.0f) * 255.99f;
    g = clamp(g, 0.0f, 1.0f) * 255.99f;
    b = clamp(b, 0.0f, 1.0f) * 255.99f;
    return (0xFFu << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b); // ARGB in register -> BGRA in memory
}
__device__ uint32_t RgbToInt_(float3 &c)
{
    return RgbToInt_(c.x, c.y, c.z);
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

    float f;

    if (c < CM_COLOR_DIST_END) {
        T dist;
        switch (t) {
        case CM_SQR_GENERIC:
            dist = MandelbrotDist<T, CM_SQR_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_CUBE_GENERIC:
            dist = MandelbrotDist<T, CM_CUBE_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_FULL_GENERIC:
            dist = MandelbrotDist<T, CM_FULL_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_BURNING_SHIP_GENERIC:
            dist = MandelbrotDist<T, CM_BURNING_SHIP_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_SQR_FLOAT:
            dist = MandelbrotDist<T, CM_SQR_FLOAT>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_SQR_DOUBLE:
            dist = MandelbrotDist<T, CM_SQR_DOUBLE>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        }
        f = float(dist / zoom);
    }
    else {
        switch (t) {
        case CM_SQR_GENERIC:
            f = MandelbrotSIter<T, CM_SQR_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_CUBE_GENERIC:
            f = MandelbrotSIter<T, CM_CUBE_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_FULL_GENERIC:
            f = MandelbrotSIter<T, CM_FULL_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_BURNING_SHIP_GENERIC:
            f = MandelbrotSIter<T, CM_BURNING_SHIP_GENERIC>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_SQR_FLOAT:
            f = MandelbrotSIter<T, CM_SQR_FLOAT>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        case CM_SQR_DOUBLE:
            f = MandelbrotSIter<T, CM_SQR_DOUBLE>()(x, y, bailout, z0_x, z0_y, iter, exponent);
            break;
        }
        f = f * (256.0f / float(iter));
    }

    float3 rgb;
    switch (c) {
    case CM_DIST_BLACK_BROWN_BLUE:
        rgb = ColorizeMandelbrot<CM_DIST_BLACK_BROWN_BLUE>(f);
        break;
    case CM_DIST_GREEN_BLUE:
        rgb = ColorizeMandelbrot<CM_DIST_GREEN_BLUE>(f);
        break;
    case CM_DIST_SNOWFLAKE:
        rgb = ColorizeMandelbrot<CM_DIST_SNOWFLAKE>(f);
        break;
    case CM_ITER_BLACK_BROWN_BLUE:
        rgb = ColorizeMandelbrot<CM_ITER_BLACK_BROWN_BLUE>(f);
    }

    image_buffer[iy * w + ix] = RgbToInt_(rgb);
}


// not implemented
template<typename T>
struct MandelbrotSIter<T, CM_SQR_FLOAT> {
    __device__ __forceinline__ T operator()(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent) const {
        return 0.0f;
    }
};
template<typename T>
struct MandelbrotSIter<T, CM_SQR_DOUBLE> {
    __device__ __forceinline__ T operator()(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent) const {
        return 0.0;
    }
};
