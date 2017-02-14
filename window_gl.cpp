#ifndef CM_NOGL

#ifdef _WIN32
    #if _MSC_VER
        #define _CRT_SECURE_NO_WARNINGS
        #define snprintf sprintf_s
        #define fprintf fprintf_s
        #define fscanf fscanf_s
    #endif
#else
    #define VK_ESCAPE 27
#endif


#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#ifndef _WIN32
#define APIENTRY
#endif

#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "timing.h"
#include "window_gl.h"
#include "kernel.h"
#include "cmdline_parser.h"
#include "bmp_output.h"

static void DrawTexture();

// buffer handles/pointers
static GLuint pbo; // this PBO (pixel buffer object) is used to connect CUDA and OpenGL
static GLuint result_texture;  // the result is copied to this OpenGL texture
static uint32_t* image_buffer; // this is the actual memory pointer to the PBO

// Window & Image
#define CM_IMAGE_FORMAT GL_BGRA
static uint32_t window_width  = 1280;
static uint32_t window_height = 720;
static GLint    maxTextureSize;

// Mandelbrot kernel parameters
static kernel_params params;

// main display loop
static void Display() {

    // clear the frame buffer
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // CUDA kernel timer
    cudaTimer t;
    t = startCudaTimer();

    // map OpenGL buffer (PBO) into CUDA address space
    cudaGLMapBufferObject((void**) &params.image_buffer, pbo);

    LaunchKernel(params);

    cudaGLUnmapBufferObject(pbo); // unmap buffer so OpenGL can use our data

    float gpu_time = stopCudaTimer(t);

    // copy the data from the PBO into our result texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, result_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, params.width, params.height, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    DrawTexture();       // draw our result texture to the back buffer
    glutSwapBuffers();   // swap the window's frame buffers
    glutPostRedisplay(); // tell glut that our window has changed
    
    static char buf[256] = { 0 };
    snprintf(buf, sizeof(buf) - 1, "CUDA-Mandelbrot - GPU: %6.3fms - x: %e | y : %e | z : %e | i : %d", gpu_time, params.centerX, params.centerY, params.zoom, params.iter);
    glutSetWindowTitle(buf);
}

// display image to the screen as textured quad
static void DrawTexture() {

    // disable lighting and depth, enable textures
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // set our view projection to orthographic, coordinates from -1,-1,-1 to 1, 1, 1
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    
    // render a screen sized rectangle
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);              // begin a quadrilateral
    glTexCoord2f(0.0, 0.0);         // bottom left
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);         // bottom right
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);         // top right
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);         // top left
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

// initialize the PBO and texture for transferring data from CUDA to OpenGL
static void InitGLBuffers() {

    size_t num_texels = params.width * params.height;
    size_t size_tex_data = sizeof(*image_buffer) * num_texels;

    // create buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW); // set buffer size

    // register this buffer object with CUDA
    cudaGLRegisterBufferObject(pbo);

    // create the texture that we use to display the result
    glGenTextures(1, &result_texture);
    glBindTexture(GL_TEXTURE_2D, result_texture);

    // set texture filtering (linear required to simulate supersampling)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // set texture size and format
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, params.width, params.height, 0, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);
}

// delete and recreate buffer
static void ResizeImageBuffer() {
    params.width  = window_width  * params.sqrtSamples;
    params.height = window_height * params.sqrtSamples;
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &result_texture);
    InitGLBuffers();
}

static void WindowResizeCallback(int width, int height) {
    // set new OpenGL view port
    glViewport(0, 0, width, height);
    window_width = width;
    window_height = height;

    ResizeImageBuffer();
}


static void WriteImageToDisk(const char* filename, bool writeLargeImage = false) {
    uint32_t bufferSize;
    uint32_t *cpuBuffer;
    int result;

    if (writeLargeImage) {
        // writes the entire image buffer to disk, possibly hundreds of MB!
        bufferSize = params.width * params.height * sizeof(*image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        cudaGLMapBufferObject((void**) &image_buffer, pbo);
        cudaMemcpy(cpuBuffer, image_buffer, bufferSize, cudaMemcpyDeviceToHost);
        cudaGLUnmapBufferObject(pbo);
        result = write_bmp(filename, params.width, params.height, (uint8_t*) cpuBuffer);
    }
    else {
        // writes the window contents to disk
        bufferSize = window_width * window_height * sizeof(*image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        glReadPixels(0, 0, window_width, window_height, GL_BGRA, GL_UNSIGNED_BYTE, cpuBuffer);
        result = write_bmp(filename, window_width, window_height, (uint8_t*) cpuBuffer);
    }

    free(cpuBuffer);
    if (result == 0) {
        printf("Wrote image to %s\n", filename);
    }
}


///////////
// INPUT //
///////////

static double zoomSpeed = 1.1;
static double startModSpeed = 0.025;
static int lmbState = GLUT_UP;
static int rmbState = GLUT_UP;
static int lastX; // last mouse X position
static int lastY; // last mouse Y position

// only called while a mouse button is down
static void MouseDragCallback(int x, int y) {
    
    // set scale so that the mouse position stays constant in the mandelbrot coordinate system
    double scale = (2.0 / double(window_height)) * params.zoom;

    if (lmbState == GLUT_DOWN) {
        params.centerX -= (x - lastX) * scale;
        params.centerY += (y - lastY) * scale;
    }
    if (rmbState == GLUT_DOWN) {
        params.z0_x += (x - lastX) * scale;
        params.z0_y -= (y - lastY) * scale;
    }
    lastX = x;
    lastY = y;
}

static void MouseClickCallback(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        lmbState = state;
        lastX = x;
        lastY = y;
    }
    if (button == GLUT_RIGHT_BUTTON) {
        rmbState = state;
        lastX = x;
        lastY = y;
    }
}

static void MouseWheelCallback(int wheel, int direction, int x, int y) {
    if (direction == -1)
        params.zoom *= zoomSpeed;
    else
        params.zoom *= 1.0 / zoomSpeed;
}

static void KeyboardCallback(unsigned char key, int x, int y) {

    int mod = glutGetModifiers();
    bool ctrlPressed = ((mod & GLUT_ACTIVE_CTRL) != 0);

    if (ctrlPressed) {
        // GLUT is really old. CTRL key modifies the ASCII key code by setting bits 5 and 6 to 0
        // See https://en.wikipedia.org/wiki/Control_key#History
        
        // NOTE: This only works for letters and also converts all of them to lower case.
        //       If I needed more accurate input I would not use GLUT at all.
        key = key | 0x60;
    }

    switch (key) {

    case VK_ESCAPE: // exit
        glutLeaveMainLoop();
        break;

    // switch mandelbrot type
    case '1':
        params.type = CM_SQR_GENERIC;
        break;
    case '2':
        params.type = CM_CUBE_GENERIC;
        break;
    case '3':
        params.type = CM_FULL_GENERIC;
        break;
    case '4':
        params.type = CM_BURNING_SHIP_GENERIC;
        break;

    // switch coloring type
    case '6':
        params.color = CM_ITER_BLACK_BROWN_BLUE;
        break;
    case '7':
        params.color = CM_DIST_SNOWFLAKE;
        break;
    case '8':
        params.color = CM_DIST_GREEN_BLUE;
        break;
    case '9':
        params.color = CM_DIST_BLACK_BROWN_BLUE;
        break;

    case 'e': // zoom out
    case '-':
        params.zoom *= zoomSpeed;
        break;
    case 'q': // zoom in
    case '+':
        params.zoom *= 1.0 / zoomSpeed;
        break;

    case 'w': // move start up
        params.z0_y += startModSpeed * params.zoom;
        break;
    case 's': // move start down
        params.z0_y -= startModSpeed * params.zoom;
        break;

    case 'a': // move start left
        params.z0_x -= startModSpeed * params.zoom;
        break;
    case 'd': // move start right
        params.z0_x += startModSpeed * params.zoom;
        break;

    case 'c': // decrease bailout radius
        params.bailout *= 1.0 / zoomSpeed;
        break;
    case 'v': // increase bailout radius
        params.bailout *= zoomSpeed;
        break;

    case ',': // decrease exponent
        params.exponent -= 0.01 * params.zoom;
        break;
    case '.': // increase exponent
        params.exponent += 0.01 * params.zoom;
        break;

    case 'k': // decrease zoom speed
        zoomSpeed -= 0.01;
        zoomSpeed = fmax(zoomSpeed, 1.01);
        break;
    case 'l': // increase zoom speed
        zoomSpeed += 0.01;
        zoomSpeed = fmax(zoomSpeed, 1.01);
        break;

    case 'f': // decrease iterations
        params.iter = (int)(params.iter * 1.0 / zoomSpeed);
        if (params.iter == 0) params.iter = 1;
        break;
    case 'g': // increase iterations
        params.iter = (int) ceil(params.iter * zoomSpeed);
        break;

    case 'b': // decrease samples per pixel
        if (params.sqrtSamples > 1) {
            params.sqrtSamples >>= 1;
            ResizeImageBuffer();
        }
        break;

    case 'n': { // increase samples per pixel
        uint32_t newSpp = params.sqrtSamples << 1;
        if (((window_width  * newSpp) < (uint32_t) maxTextureSize) &&
            ((window_height * newSpp) < (uint32_t) maxTextureSize)) {

            params.sqrtSamples = newSpp;
            ResizeImageBuffer();
        }
    } break;

    case 'r': { // reset params
        params = kernel_params();
        ResizeImageBuffer();
    } break;

    case 'i': { // write image and parameters to disk
        static char filename[64];
        static uint32_t prefix = uint32_t(getTime() ^ (getTime() >> 32));
        static uint32_t fileCounter = 0;
        int len = snprintf(filename, sizeof(filename) - 1, "output_%u_%u.bmp", prefix, fileCounter++);

        WriteImageToDisk(filename, ctrlPressed);

        // replace ".bmp" with ".par"
        assert(len < sizeof(filename));
        filename[len - 3] = 'p';
        filename[len - 2] = 'a';
        filename[len - 1] = 'r';
        WriteParamsToDisk(filename, params);
    } break;

    case 'p': { // read or write parameters from/to disk
        if (ctrlPressed) {
            if (ReadParamsFromDisk("input.par", &params)) {
                ResizeImageBuffer();
            }            
        }
        else {
            WriteParamsToDisk("input.par", params);
        }
    } break;
    }
}

static void APIENTRY PrintDebugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam) {
    printf("%s\n", message);
}

void GLWindowMain(int argc, char *argv[], const kernel_params& p) {

    // create OpenGL context and window
#if _DEBUG
    glutInitContextFlags(GLUT_DEBUG);
#endif
    glutInit(&argc, (char**) argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA-Mandelbrot");
    
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object GL_EXT_framebuffer_object ")) {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        return;
    }

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    params = p;
    params.width  = window_width  * p.sqrtSamples;
    params.height = window_height * p.sqrtSamples;

    InitGLBuffers();

    // register callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(WindowResizeCallback);
    glutKeyboardFunc(KeyboardCallback);
    glutMouseWheelFunc(MouseWheelCallback);
    glutMouseFunc(MouseClickCallback);
    glutMotionFunc(MouseDragCallback);
    
#if _DEBUG
    printf("  VENDOR: %s\n", (const char *) glGetString(GL_VENDOR));
    printf(" VERSION: %s\n", (const char *) glGetString(GL_VERSION));
    printf("RENDERER: %s\n", (const char *) glGetString(GL_RENDERER));

    // register debug callback (either OpenGL 4.3 or GL_ARB_debug_output is required)
    if (glewIsSupported("GL_VERSION_4_3")) {
        glDebugMessageCallback((GLDEBUGPROC) PrintDebugMessage, NULL);
        // set debug output to be called in the same thread so we can set a breakpoint
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }
    else if (glewIsSupported("GL_ARB_debug_output")) {
        glDebugMessageCallbackARB((GLDEBUGPROC) PrintDebugMessage, NULL);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    }
#endif

    cudaProfilerStart();

    // start rendering main-loop
    glutMainLoop();
}

#endif
