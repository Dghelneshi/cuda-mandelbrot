#ifndef CM_NOGL

#ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #define snprintf sprintf_s
#else
    #define VK_ESCAPE 27
#endif


#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <math.h>
#include "stdint.h"
#include "timing.h"
#include "window_gl.h"
#include "main.h"
#include "kernel.h"

static void displayTexture();

static GLuint pbo; // this PBO is used to connect CUDA and OpenGL
static GLuint result_texture; // the result is copied to this OpenGL texture
static float4* compute_buffer;

static uint32_t window_width = 1200;
static uint32_t window_height = 800;
static uint32_t image_width = window_width;
static uint32_t image_height = window_height;

static double speed = 0.1;
static double zoomSpeed = 1.1;
static double maxlen2 = 1024.0;
static pos_t pos;


static void display() {
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // CUDA kernel timer
    static cudaTimer t = { 0 };
    float gpu_time = 0.0f;
    if (t.start != 0) { // we made a round trip
        gpu_time = stopCudaTimer(t);
    }
    t = startCudaTimer();

    checkCudaErrors(cudaGLMapBufferObject((void**) &compute_buffer, pbo));

    launchKernel(compute_buffer, image_width, image_height, pos, maxlen2);

    checkCudaErrors(cudaGLUnmapBufferObject(pbo));

    // download texture from destination PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, result_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    SDK_CHECK_ERROR_GL();

    displayTexture();
    glutSwapBuffers();
    glutPostRedisplay();

    static char buf[256] = { 0 };
    snprintf(buf, sizeof(buf) - 1, "Mandelbrot Test - GPU: %6.3fms - x: %e | y : %e | z : %e | m : %f", gpu_time, pos.centerX, pos.centerY, pos.zoom, maxlen2);
    glutSetWindowTitle(buf);
}

static void keyboard(unsigned char key, int /*x*/, int /*y*/) {

    double move = speed * pos.zoom;

    switch (key) {
    case VK_ESCAPE:
        terminate(0);
        break;
    case 'e':
        pos.zoom *= zoomSpeed;
        break;
    case 'q':
        pos.zoom *= 1.0 / zoomSpeed;
        break;
    case 'w':
        pos.centerY += move;
        break;
    case 's':
        pos.centerY -= move;
        break;
    case 'a':
        pos.centerX -= move;
        break;
    case 'd':
        pos.centerX += move;
        break;
    case 'v':
        maxlen2 *= zoomSpeed;
        break;
    case 'c':
        maxlen2 *= 1.0 / zoomSpeed;
        break;
    case 'j':
        speed -= 0.01;
        break;
    case 'u':
        speed += 0.01;
        break;
    case 'k':
        zoomSpeed -= 0.01;
        break;
    case 'i':
        zoomSpeed += 0.01;
        break;
    case 'r':
        pos.centerX = -0.5;
        pos.centerY = 0.0;
        pos.zoom = 1.0;
        maxlen2 = 1024.0;
        break;
    }
    zoomSpeed = fmax(zoomSpeed, 1.01);
    speed = fmax(speed, 0.01);
}


// display image to the screen as textured quad
static void displayTexture() {
    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    SDK_CHECK_ERROR_GL();
}

static void initGLBuffers() {

    // initialize the PBO for transferring data from CUDA to openGL
    size_t num_texels = image_width * image_height;
    size_t size_tex_data = sizeof(float4) * num_texels;
    void* data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGLRegisterBufferObject(pbo));
    SDK_CHECK_ERROR_GL();

    // create the texture that we use to visualize the ray-tracing result
    glGenTextures(1, &result_texture);
    glBindTexture(GL_TEXTURE_2D, result_texture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, image_width, image_height, 0, GL_RGBA, GL_FLOAT, NULL);
    SDK_CHECK_ERROR_GL();
}

// Callback function called by GLUT when window size changes
static void reshape(int width, int height) {
    // TODO: resize image buffer (without crashing!)
    window_width = width;
    window_height = height;
    // Set OpenGL view port and camera
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0f, (double) width / height, 0.1, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


bool initGLWindow(int argc, const char *argv[]) {

    // Create GL context
    glutInit(&argc, (char**) argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Mandelbrot Test");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object GL_EXT_framebuffer_object ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

    initGLBuffers();

    // register callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    return true;
}

#endif