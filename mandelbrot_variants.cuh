#include <vector_functions.h>
#include "helper_math.h"
#include "double2_inline.h"

// NOTE: changes in this file will not automatically trigger a rebuild in Visual Studio!

// Experiment results:
// 1. complex<T>/vec2<T> datastructures are slower than any other method
// 2. float2/double2 are faster than any other method
// 3. individual floats/doubles are just slightly slower than float2/double2
// 4. it's either impossible or at least extremely ugly to "typedef" a vec2<T> as float2 and double2 respectively

template<typename T>
struct vec2 {
    T x;
    T y;
    __device__ vec2(T x, T y) : x(x), y(y) {}
    __device__ vec2 operator*(T b) {
        return vec2(x*b, y*b);
    }
    __device__ friend static vec2 operator*(T a, vec2 b) {
        return vec2(a*b.x, a*b.y);
    }
    __device__ vec2 operator+(vec2 b) {
        return vec2(x + b.x, y + b.y);
    }
};

template<typename T>
__device__ static T length(vec2<T> v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

// complex power function
template<typename T>
__device__ static vec2<T> cpow(vec2<T> c, T exponent)
{
    T cAbs = length(c);
    vec2<T> cLog = vec2<T>(log(cAbs), atan2(c.y, c.x + (T)1e-5));
    vec2<T> cMul = exponent*cLog;
    T expReal = exp(cMul.x);
    return vec2<T>(expReal*cos(cMul.y), expReal*sin(cMul.y));
}


template<typename T>
static __device__ T MandelbrotFullGeneric(T x, T y, T maxlen2, T startX, T startY, int iter, T e) {

    vec2<T> c(x,y);
    vec2<T> z(startX, startY);
    vec2<T> dz((T)0.0, (T)0.0); // derivative z'
    T len2;

    int i;
    for (i = 0; i < iter; i++) {
        // z' = e * z^(e-1) * z' + 1
        vec2<T> chain = e * cpow(z, e - (T)1.0);
        dz = vec2<T>(chain.x*dz.x - chain.y*dz.y, chain.x*dz.y + chain.y*dz.x) + vec2<T>((T)1.0, (T)0.0);

        // z = z^e + c
        z = cpow(z, e) + c;

        len2 = z.x*z.x + z.y*z.y;
        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    // NOTE: log(sqrt(x)) = 0.5 * log(x)
    T dzlen2 = dz.x*dz.x + dz.y*dz.y;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);

    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

template<typename T>
static __device__ T MandelbrotSquareGeneric(T x, T y, T maxlen2, T startX, T startY, int iter) {

    T cx = x;
    T cy = y;
    T zx = startX;
    T zy = startY;
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
        zx = zx*zx - zy*zy + cx;
        zy = (T)2.0 * zx_*zy + cy;

        len2 = zx*zx + zy*zy;
        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    T dzlen2 = dzx*dzx + dzy*dzy;
    T d = (T)0.5 * sqrt(len2 / dzlen2) * log(len2);
    
    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

static __device__ double MandelbrotSquareDouble(double x, double y, double maxlen2, double startX, double startY, int iter) {

    double2 c = make_double2(x, y);
    double2 z = make_double2(startX, startY);
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
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    double d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);

    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

static __device__ float MandelbrotSquareFloat(float x, float y, float maxlen2, float startX, float startY, int iter) {

    float2 c = make_float2(x, y);
    float2 z = make_float2(startX, startY);
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
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    float d = 0.5f * sqrt(len2 / dot(dz, dz)) * log(len2);

    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}

static __device__ double MandelbrotCubeDouble(double x, double y, double maxlen2, double startX, double startY, int iter) {

    double2 c = make_double2(x, y);
    double2 z = make_double2(startX, startY);
    double2 dz = make_double2(0.0, 0.0); // derivative z'
    double2 z2;
    double len2 = 0.0;

    int i;
    for (i = 0; i < iter; i++) {

        // z' = 3 * z^2 * z' + 1
        z2 = make_double2(z.x*z.x - z.y*z.y, 2.0f*z.x*z.y);
        dz = 3.0 * make_double2(z2.x * dz.x - z2.y * dz.y, z2.x * dz.y + z2.y * dz.x) + make_double2(1.0, 0.0);

        // z = z^3 + c
        z = make_double2(z2.x*z.x - z2.y*z.y, z2.x*z.y + z2.y*z.x) + c;

        len2 = dot(z, z);

        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    double d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);

    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}



static __device__ float MandelbrotCubeFloat(float x, float y, float maxlen2, float startX, float startY, int iter) {

    float2 c = make_float2(x, y);
    float2 z = make_float2(startX, startY);
    float2 dz = make_float2(0.0, 0.0); // derivative z'
    float2 z2; // z^2
    float len2 = 0.0;

    int i;
    for (i = 0; i < iter; i++) {

        // z' = 3*z^2*z' + 1
        z2 = make_float2(z.x*z.x - z.y*z.y, 2.0f*z.x*z.y);
        dz = 3.0 * make_float2(z2.x * dz.x - z2.y * dz.y, z2.x * dz.y + z2.y * dz.x) + make_float2(1.0, 0.0);

        // z = z^3 + c
        z = make_float2(z2.x*z.x - z2.y*z.y, z2.x*z.y + z2.y*z.x) + c;

        len2 = dot(z, z);

        // if z is too far from the origin, assume divergence
        if (len2 > maxlen2) break;
    }

    // distance	estimation
    // d(c) = |z|*log|z|/|z'|
    float d = 0.5 * sqrt(len2 / dot(dz, dz)) * log(len2);

    return (i == iter) ? 0.0 : d; // estimate can be wrong inside blobs, so use iteration count as well
}



enum cm_variants {
    SQR_GENERIC,
    SQR_FLOAT,
    SQR_DOUBLE,
    CUBE_FLOAT,
    CUBE_DOUBLE,
    SQR_FULL_GENERIC,
};

// This template function is essentially a compile-time switch statement for the different mandelbrot variants

template<typename T, cm_variants V>
__device__ __forceinline__ T Mandelbrot(T x, T y, T maxlen2, T startX, T startY, int iter, T exponent);

template<> float Mandelbrot<float, SQR_GENERIC>(float x, float y, float maxlen2, float startX, float startY, int iter, float exponent) {
    return MandelbrotSquareGeneric<float>(x, y, maxlen2, startX, startY, iter);
}
template<> double Mandelbrot<double, SQR_GENERIC>(double x, double y, double maxlen2, double startX, double startY, int iter, double exponent) {
    return MandelbrotSquareGeneric<double>(x, y, maxlen2, startX, startY, iter);
}
template<> float Mandelbrot<float, SQR_FULL_GENERIC>(float x, float y, float maxlen2, float startX, float startY, int iter, float exponent) {
    return MandelbrotFullGeneric<float>(x, y, maxlen2, startX, startY, iter, exponent);
}
template<> double Mandelbrot<double, SQR_FULL_GENERIC>(double x, double y, double maxlen2, double startX, double startY, int iter, double exponent) {
    return MandelbrotFullGeneric<double>(x, y, maxlen2, startX, startY, iter, exponent);
}
template<> float Mandelbrot<float, SQR_FLOAT>(float x, float y, float maxlen2, float startX, float startY, int iter, float exponent) {
    return MandelbrotSquareFloat(x, y, maxlen2, startX, startY, iter);
}
template<> double Mandelbrot<double, SQR_DOUBLE>(double x, double y, double maxlen2, double startX, double startY, int iter, double exponent) {
    return MandelbrotSquareDouble(x, y, maxlen2, startX, startY, iter);
}
template<> float Mandelbrot<float, CUBE_FLOAT>(float x, float y, float maxlen2, float startX, float startY, int iter, float exponent) {
    return MandelbrotCubeFloat(x, y, maxlen2, startX, startY, iter);
}
template<> double Mandelbrot<double, CUBE_DOUBLE>(double x, double y, double maxlen2, double startX, double startY, int iter, double exponent) {
    return MandelbrotCubeDouble(x, y, maxlen2, startX, startY, iter);
}

