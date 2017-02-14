#include <vector_functions.hpp>
#include <device_functions.hpp>
#include "helper_math.h"
#include "double2_inline.h"
#include <limits>

// NOTE: changes in this file will not automatically trigger a rebuild in Visual Studio!

// Experimental results:
// 1. complex<T>/vec2<T> datastructures are slower than any other method
// 2. float2/double2 are faster than any other method
// 3. individual floats/doubles are just slightly slower than float2/double2
// 4. it's either impossible or at least extremely ugly to "typedef" a vec2<T> as float2 and double2 respectively

// templated version of sincos()
template<typename T>
__device__ __forceinline__ void sincos(T a, T *sp, T *cp);
template<>
__device__ __forceinline__ void sincos<float>(float a, float *sp, float *cp) {
    sincosf(a, sp, cp);
}
template<>
__device__ __forceinline__ void sincos<double>(double a, double *sp, double *cp) {
    sincos(a, sp, cp);
}

// complex power function (real exponent)
// z = x + iy
// z = e ^ (log(r) + i*theta)         [Cartesian to polar coordinates]
// log(z) = log(r) + i*theta
// r = sqrt(x^2 + y^2)
// theta = atan2(y,x))
// z^n = e ^ (n * log(z))
//     = e ^ (n * (log(r) + i*theta))
//     = e^(n*log(r)) * e^(n*i*theta)
//     = e^(n*log(r)) * (cos(n*theta) + i*sin(n*theta))   [polar to Cartesian]
//     =     r^n      * (             ...             )   [unused, pow() is too slow]
//
// log(r) = log(sqrt(x^2 + y^2)) = 1/2 * log(x^2 + y^2)   [avoid sqrt]

template<typename T>
__device__ __forceinline__ static void cpow(T x, T y, T& xres, T& yres, T n)
{
    T r2 = x*x + y*y;
    T theta = atan2(y, x);
    T mul = exp((T)0.5 * log(r2) * n);
    T s, c;
    sincos<T>(theta * n, &s, &c);
    xres = mul * c;
    yres = mul * s;
    
}

template<typename T>
__device__ __forceinline__ static void cpow(T &x, T &y, T n)
{
    T r2 = x*x + y*y;
    T theta = atan2(y, x);
    T mul = exp((T)0.5 * log(r2) * n);
    T s, c;
    sincos<T>(theta * n, &s, &c);
    x = mul * c;
    y = mul * s;
}


// Generic Mandelbrot with arbitrary exponent
// returns distance estimation
// z = z^n + c
template<typename T>
static __device__ T MandelbrotDistFullGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter, T n) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T dzx = (T)0.0; // derivative z'
    T dzy = (T)0.0;
    T len2 = (T)0.0;
    T chainx; // z^(n-1)
    T chainy;

    int i;
    for (i = 0; i < iter; i++) {

        // z' = n * z^(n-1) * z' + 1
        cpow(zx, zy, chainx, chainy, n - (T)1.0);
        chainx *= n;
        chainy *= n;
        T dzx_ = dzx;
        dzx = chainx*dzx - chainy*dzy + 1.0;
        dzy = chainx*dzy + chainy*dzx_;

        // z = z^n + c
        cpow(zx,zy,n);
        zx += cx;
        zy += cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    // log(sqrt(r)) = 0.5 * log(r)
    T dzlen2 = dzx*dzx + dzy*dzy;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);

    return (i == iter) ? (T)0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

// Generic Mandelbrot with exponent = 2
// returns distance estimation
// z = z^2 + c
template<typename T>
static __device__ T MandelbrotDistSquareGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T dzx = (T)0.0; // derivative z'
    T dzy = (T)0.0;
    T len2 = (T)0.0;

    int i;
    for (i = 0; i < iter; i++) {
        // z' = 2*z*z' + 1
        T dzx_ = dzx;
        dzx = (T)2.0 * (zx*dzx - zy*dzy) + (T)1.0;
        dzy = (T)2.0 * (zx*dzy + zy*dzx_);

        // z = z^2 + c
        T zx_ = zx;
        zx =  zx*zx - zy*zy  + cx;
        zy = (T)2.0 * zx_*zy + cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    T dzlen2 = dzx*dzx + dzy*dzy;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);
    
    return (i == iter) ? (T)0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

// Generic Mandelbrot with exponent = 3
// returns distance estimation
// z = z^3 + c
template<typename T>
static __device__ T MandelbrotDistCubeGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T dzx = (T)0.0; // derivative z'
    T dzy = (T)0.0;
    T len2 = (T)0.0;
    T z2x; // z^2
    T z2y;

    int i;
    for (i = 0; i < iter; i++) {

        // z^2
        z2x =  zx*zx - zy*zy;
        z2y = T(2.0) * zx*zy;

        // z' = 3 * z^2 * z' + 1
        T dzx_ = dzx;
        dzx = T(3.0) * (z2x*dzx - z2y*dzy) + T(1.0);
        dzy = T(3.0) * (z2x*dzy + z2y*dzx_);
        
        // z = z^3 + c
        T zx_ = zx;
        zx = z2x*zx - z2y*zy  + cx;
        zy = z2x*zy + z2y*zx_ + cy;

        len2 = zx*zx + zy*zy;

        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    T dzlen2 = dzx*dzx + dzy*dzy;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);

    return (i == iter) ? (T)0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}


// "Burning Ship" with exponent = 2
// returns distance estimation
// z = (|x| + i|y|)^2 - c
template<typename T>
static __device__ float BurningShipDistSquareGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T len2 = (T)0.0;
    T dzx = (T)0.0; // derivative z'
    T dzy = (T)0.0;

    int i;
    for (i = 0; i < iter; i++) {
        // z' = 2*z*z' + 1
        // not sure if correct?
        T dzx_ = dzx;
        dzx = (T)2.0 * (zx*dzx - zy*dzy) + (T)1.0;
        dzy = (T)2.0 * (zx*dzy + zy*dzx_);

        // z = (|x| + i|y|)^2 - c
        T zx_ = zx;
        zx = zx*zx - zy*zy - cx;
        zy = (T)2.0 * fabs(zx_*zy) - cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    T dzlen2 = dzx*dzx + dzy*dzy;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);

    return (i == iter) ? 0.0f : d; // estimate can be wrong inside blobs, so use iteration count as well
}




/////////////////////////////
// Smooth Iteration Count  //
/////////////////////////////

// Generic Mandelbrot with arbitrary exponent
// returns smooth iteration count
// z = z^n + c
template<typename T>
static __device__ float MandelbrotFullGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter, T n) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T len2 = T(0.0);

    int i;
    for (i = 0; i < iter; i++) {

        // z = z^n + c
        cpow(zx, zy, n);
        zx += cx;
        zy += cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // smooth iteration count
    float si = float(i) - log2(log2(float(len2)) / log2(float(bailout))) / log2(float(n));

    return (i == iter) ? NAN : si; // prevent artifacts inside blobs
}


// Generic Mandelbrot with exponent = 2
// returns smooth iteration count
// z = z^2 + c
template<typename T>
static __device__ float MandelbrotSquareGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T len2 = (T)0.0;

    int i;
    for (i = 0; i < iter; i++) {

        // z = z^2 + c
        T zx_ = zx;
        zx = zx*zx - zy*zy + cx;
        zy = (T)2.0 * zx_*zy + cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // smooth iteration count
    float si = float(i) - log2(log2(float(len2)) / log2(float(bailout)));

    return (i == iter) ? NAN : si; // prevent artifacts inside blobs
}

// Generic Mandelbrot with exponent = 3
// returns smooth iteration count
// z = z^3 + c
template<typename T>
static __device__ T MandelbrotCubeGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T len2;
    T z2x; // z^2
    T z2y;

    int i;
    for (i = 0; i < iter; i++) {

        // z^2
        z2x = zx*zx - zy*zy;
        z2y = T(2.0) * zx*zy;

        // z = z^3 + c
        T zx_ = zx;
        zx = z2x*zx - z2y*zy  + cx;
        zy = z2x*zy + z2y*zx_ + cy;

        len2 = zx*zx + zy*zy;

        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // smooth iteration count
    float si = float(i) - log2(log2(float(len2)) / log2(float(bailout))) / log2(3.0f);

    return (i == iter) ? NAN : si; // prevent artifacts inside blobs
}


// "Burning Ship" with exponent = 2
// returns smooth iteration count
// z = (|x| + i|y|)^2 - c
template<typename T>
static __device__ float BurningShipSquareGeneric(T x, T y, T bailout, T z0_x, T z0_y, int iter) {

    T cx = x;
    T cy = y;
    T zx = z0_x;
    T zy = z0_y;
    T len2 = (T)0.0;

    int i;
    for (i = 0; i < iter; i++) {

        // z = (|x| + i|y|)^2 - c
        T zx_ = zx;
        zx =  zx*zx - zy*zy        - cx;
        zy = (T)2.0 * fabs(zx_*zy) - cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > bailout) break;
    }

    // smooth iteration count
    float si = float(i) - log2(log2(float(len2)) / log2(float(bailout)));

    return (i == iter) ? NAN : si; // prevent artifacts inside blobs
}


////////////////////////
// Coloring Functions //
////////////////////////

template<cm_colors C>
__device__ __forceinline__ float3 ColorizeMandelbrot(float f);

// color by distance

template<>
__device__ float3 ColorizeMandelbrot<CM_DIST_SNOWFLAKE>(float dist) {
    // honestly, I have no idea how this one works
    dist = clamp(12.0f * dist, 0.0f, 256.0f);
    dist = rcbrt(rsqrt(dist)); // dist^(1/6)
    dist = (256.0f - dist) + 4.0f;

    return make_float3(cos(3.0f + dist), cos(3.6f + dist), cos(4.0f + dist));
}

template<>
__device__ float3 ColorizeMandelbrot<CM_DIST_GREEN_BLUE>(float dist) {
    dist = clamp(dist, 0.0f, 1.0f);
    dist = rcbrt(rsqrt(dist)); // dist^(1/6)

    return 0.3f + 0.7f * make_float3(-1.0f, cospif(dist), sinpif(dist));
}

template<>
__device__ float3 ColorizeMandelbrot<CM_DIST_BLACK_BROWN_BLUE>(float dist) {
    dist = clamp(dist, 0.0f, 1.0f);
    dist = (1.0f - dist);
    if (dist >= 0.9999f) dist = 0.0f;
    dist = pow(dist, 8.0f) * 4.0f;
    return 0.3f + 0.7f * make_float3(cos(3.0f + dist), cos(3.6f + dist), cos(4.0f + dist));
}

// color by iterations

template<>
__device__ float3 ColorizeMandelbrot<CM_ITER_BLACK_BROWN_BLUE>(float i) {
    float3 rgb;
    if (isnan(i)) {
        rgb = make_float3(0.0);
    }
    else {
        rgb = 0.5f + 0.5f * make_float3(cos(3.0f + i*0.15f), cos(3.6f + i*0.15f), cos(4.0f + i*0.15f));
    }
    return rgb;
}

//////////////////////////

// unfortunately, partial specialization of functions is not supported by C++.
// this is why we need to write all of this nonsense to instantiate the float/double variants explicitly.

template<typename T, cm_type V>
__device__ __forceinline__ T MandelbrotDist(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent);

template<> float MandelbrotDist<float, CM_SQR_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotDistSquareGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> double MandelbrotDist<double, CM_SQR_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotDistSquareGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotDist<float, CM_CUBE_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotDistCubeGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> double MandelbrotDist<double, CM_CUBE_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotDistCubeGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotDist<float, CM_FULL_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotDistFullGeneric<float>(x, y, bailout, z0_x, z0_y, iter, exponent);
}
template<> double MandelbrotDist<double, CM_FULL_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotDistFullGeneric<double>(x, y, bailout, z0_x, z0_y, iter, exponent);
}
template<> float MandelbrotDist<float, CM_BURNING_SHIP_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return BurningShipDistSquareGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> double MandelbrotDist<double, CM_BURNING_SHIP_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return BurningShipDistSquareGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}

template<typename T, cm_type V>
__device__ __forceinline__ float MandelbrotSIter(T x, T y, T bailout, T z0_x, T z0_y, int iter, T exponent);

template<> float MandelbrotSIter<float, CM_SQR_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotSquareGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotSIter<double, CM_SQR_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotSquareGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotSIter<float, CM_FULL_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotFullGeneric<float>(x, y, bailout, z0_x, z0_y, iter, exponent);
}
template<> float MandelbrotSIter<double, CM_FULL_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotFullGeneric<double>(x, y, bailout, z0_x, z0_y, iter, exponent);
}
template<> float MandelbrotSIter<float, CM_CUBE_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return MandelbrotCubeGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotSIter<double, CM_CUBE_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return MandelbrotCubeGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotSIter<float, CM_BURNING_SHIP_GENERIC>(float x, float y, float bailout, float z0_x, float z0_y, int iter, float exponent) {
    return BurningShipSquareGeneric<float>(x, y, bailout, z0_x, z0_y, iter);
}
template<> float MandelbrotSIter<double, CM_BURNING_SHIP_GENERIC>(double x, double y, double bailout, double z0_x, double z0_y, int iter, double exponent) {
    return BurningShipSquareGeneric<double>(x, y, bailout, z0_x, z0_y, iter);
}

// unused variants for performance comparisons
#include "mandelbrot_unused.cuh"

