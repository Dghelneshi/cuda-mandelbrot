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
#ifdef _WIN32
    #include <GL/wglew.h>
#else
    #include <GL/glxew.h>
    #define APIENTRY // this should be correctly defined in glew.h, but apparently not?
#endif

#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "timing.h"
#include "window_gl.h"
#include "kernel.h"
#include "cmdline_parser.h"
#include "bmp_output.h"


static void DrawResultAccumulateLanczos();
static void DrawHUD();
static void GenerateSampleDistribution();
static void SetVSync(int i);
static void DrawFullscreenTexture(float shiftX, float shiftY);

// macro for continuous printing into the same static buffer
#define SNPRINTF_CONT(buf, c, format, ...)  ((c) += snprintf((buf) + (c), sizeof(buf) - ((c) + 1), (format), ##__VA_ARGS__))


// buffer handles/pointers
static GLuint pbo; // this PBO (pixel buffer object) is used to connect CUDA and OpenGL
static GLuint result_texture;  // the kernel result is copied from the PBO to this texture

// Window & Image
#define CM_IMAGE_FORMAT GL_BGRA
static GLint maxTextureSize;
static uint32_t textureWidth;
static uint32_t textureHeight;
static bool vsync = true;

// sample distribution for image downscaling (z is Lanczos weight)
static float3 *sampleDist = nullptr;
static float totalLanczosWeight = 0.0f;

// this allows us to toggle between the CUDA implementation of Lanczos
static bool useCudaLanczos = true;
static uint32_t *offscreenBuffer = nullptr;

// states for hud/help text on screen
static bool hudState = true;
static bool helpState = true;

// Mandelbrot kernel parameters
static kernel_params params;


// TODO: code complexity is starting to get out of hand, especially due to CUDA vs GL Lanczos
// (?)   since that adds another buffer and changes their meaning/size.


// main display loop
static void DisplayCallback() {

    // clear the frame buffer
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // CUDA kernel timer
    float kernel_time, fractal_time, lanczos_time = 0.0f;
    cuda_timer t = StartCudaTimer();

    if (useCudaLanczos && !(params.sqrtSamples == 1)) {
        // use offscreen buffer to render fractal, then map PBO and resample into that
        params.image_buffer = offscreenBuffer;
        uint32_t *frameBuffer;
        cudaGLMapBufferObject((void**) &frameBuffer, pbo); // map this a bit earlier than needed to reduce overhead

        LaunchFractalKernel(params);
        fractal_time = StopCudaTimer(t);

        t = StartCudaTimer();
        LaunchResamplingKernel(offscreenBuffer, frameBuffer, params.imageWidth, params.imageHeight, params.sqrtSamples);

        cudaGLUnmapBufferObject(pbo);
        lanczos_time = StopCudaTimer(t);
    }
    else {
        // map OpenGL buffer (PBO) into CUDA address space
        cudaGLMapBufferObject((void**) &params.image_buffer, pbo);
        LaunchFractalKernel(params);
        cudaGLUnmapBufferObject(pbo); // unmap buffer so OpenGL can use our data
        fractal_time = StopCudaTimer(t);
    }
    kernel_time = fractal_time + lanczos_time;


    // copy the data from the PBO into our result texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, result_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw our result texture to the back buffer
    if (useCudaLanczos || (params.sqrtSamples == 1)) {
        DrawFullscreenTexture(0.0f, 0.0f);
    }
    else {
        DrawResultAccumulateLanczos();
    }
    DrawHUD();           // draw some helpful text on the screen
    glutSwapBuffers();   // swap the window's frame buffers
    glutPostRedisplay(); // tell glut that our window has changed

    // total frame time counter
    // note that if VSync is on, there is no way of finding out the actual overhead
    static uint64_t tStart = GetTime();
    uint64_t tEnd = GetTime();
    float time = TimeDelta(tStart, tEnd);
    tStart = tEnd;
    
    // display some performance stuff in window title
    static char buf[512] = { 0 };
    int c = 0;
    SNPRINTF_CONT(buf, c, "CUDA-Mandelbrot - Fractal: %8.3fms", fractal_time);

    if (useCudaLanczos)
        SNPRINTF_CONT(buf, c, " - CUDA Lanczos: %7.3fms", lanczos_time);
    else
        SNPRINTF_CONT(buf, c, " - OpenGL Lanczos: [included in Overhead]");

    SNPRINTF_CONT(buf, c, " - Total Frame Time: %8.3fms", time);

    if (vsync)
        SNPRINTF_CONT(buf, c, " (VSynced)");
    else
        SNPRINTF_CONT(buf, c, " (Overhead %7.3fms)", time - kernel_time);

    glutSetWindowTitle(buf);
}


static void DrawFullscreenTexture(float shiftX, float shiftY) {
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
        
    // shift the projection by sub-pixel offsets to accumulate multiple samples per pixel
    glTranslatef(shiftX, shiftY, 0.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Render a screen sized rectangle with the result texture on it.
    // Note: Actually slightly larger than screen size just in case.
    // We might sample outside the screen with our projection shift (*shouldn't* happen, but am not 100% certain)
    glBegin(GL_QUADS);              // begin a quadrilateral
    glTexCoord2f(-0.5, -0.5);       // bottom left
    glVertex3f(-2.0, -2.0, 0.5);
    glTexCoord2f(1.5, -0.5);        // bottom right
    glVertex3f(2.0, -2.0, 0.5);
    glTexCoord2f(1.5, 1.5);         // top right
    glVertex3f(2.0, 2.0, 0.5);
    glTexCoord2f(-0.5, 1.5);        // top left
    glVertex3f(-2.0, 2.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

// display image to the screen as textured quad
// we draw the texture multiple times with sub-pixel offsets and accumulate the results
// this allows us to get multiple texture samples per pixel for anti-aliasing
static void DrawResultAccumulateLanczos() {

    uint32_t spp = params.sqrtSamples * params.sqrtSamples;
    for (uint32_t i = 0; i < spp; i++) {

        // get the offsets and weight from our generated sample distribution
        float shiftX = 2.0f * sampleDist[i].x / float(params.imageWidth);
        float shiftY = 2.0f * sampleDist[i].y / float(params.imageHeight);
        float weight = sampleDist[i].z; // Lanczos weight

        // draw a screen sized texture of our image
        DrawFullscreenTexture(shiftX, shiftY);

        // accumulate the rendering result
        // using unscaled weights and dividing at the end seems to be better at higher sample counts
        float div = (spp >= 49) ? weight : (weight / totalLanczosWeight);
        glAccum(i ? GL_ACCUM : GL_LOAD, div);
    }
    // draw the accumulated result on screen
    float div = (spp >= 49) ? (1.0f / totalLanczosWeight) : 1.0f;
    glAccum(GL_RETURN, div);
}

static void DrawHUD() {

    static const char *helpText[] {
        "   H: Toggle this help text (CTRL for HUD)",
        "   R: Reset parameters",
        " Q/E: Zoom in/out (also mousewheel)",
        " LMB: Left mouse to drag the screen",
        " 1-4: Change fractal type",
        " 6-9: Change coloring",
        " C/V: Change bailout radius",
        " B/N: Change samples per pixel",
        " ,/.: Change exponent (type 3 only)"
        " F/G: Change iteration count",
        "WASD: Change z0 (also right mouse drag)",
        "   P: Write parameters to disk (CTRL to load)",
        "   I: Write image to disk (CTRL to write supersampling buffer [large])",
        " J/K: Change zoom speed",
        "   L: Toggle Lanczos downsampling implementation",
        "   X: Toggle VSync",
    };

    static char buf[11][64];
    snprintf(buf[ 0], 64, "   W: %u",    params.imageWidth);
    snprintf(buf[ 1], 64, "   H: %u",    params.imageHeight);
    snprintf(buf[ 2], 64, "   X: %.15e", params.centerX);
    snprintf(buf[ 3], 64, "   Y: %.15e", params.centerY);
    snprintf(buf[ 4], 64, "   Z: %e",    params.zoom);
    snprintf(buf[ 5], 64, "   I: %u",    params.iter);
    snprintf(buf[ 6], 64, "   E: %.15f", params.exponent);
    snprintf(buf[ 7], 64, "   B: %.3f",  params.bailout);
    snprintf(buf[ 8], 64, "z0_x: %.15e", params.z0_x);
    snprintf(buf[ 9], 64, "z0_y: %.15e", params.z0_y);
    snprintf(buf[10], 64, " spp: %u",    params.sqrtSamples * params.sqrtSamples);

    if (helpState) {
        int helpCount = sizeof(helpText) / sizeof(*helpText);
        for (int i = 0; i < helpCount; ++i) {
            glWindowPos2i(12, 12 * (helpCount - i));
            glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*) helpText[i]);
        }
    }
    if (hudState) {
        for (int i = 0; i < sizeof(buf) / sizeof(*buf); ++i) {
            glWindowPos2i(12, params.imageHeight - 12 * (i + 1) - 10);
            glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*) buf[i]);
        }
    }
}


// initialize the PBO and texture for transferring data from CUDA to OpenGL
static void InitBuffers() {

    size_t bufferSize = size_t(params.bufferWidth) * size_t(params.bufferHeight) * sizeof(*params.image_buffer);

    if (useCudaLanczos && !(params.sqrtSamples == 1)) {
        // if we use CUDA for downsampling, we render to an offscreen buffer first and then resample into the PBO
        textureWidth = params.imageWidth;
        textureHeight = params.imageHeight;
        cudaMalloc(&offscreenBuffer, bufferSize); // mallocarray malloc2d?
    }
    else {
        textureWidth = params.bufferWidth;
        textureHeight = params.bufferHeight;
    }

    // create buffer object
    size_t pboSize = size_t(textureWidth) * size_t(textureHeight) * sizeof(*params.image_buffer);
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, pboSize, NULL, GL_DYNAMIC_DRAW); // set buffer size

    // register this buffer object with CUDA
    cudaGLRegisterBufferObject(pbo);

    // create the texture that we use to display the result
    glGenTextures(1, &result_texture);
    glBindTexture(GL_TEXTURE_2D, result_texture);

    // disable texture filtering (our Lanczos resampling already handles this)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // set wrap mode to something sensible just in case we need it
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);

    // set texture size and format
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);

    if (params.sqrtSamples > 1) {
        // generate sample distribution for image downscaling
        // Note: technically only needed when sqrtSamples changes, but doesn't hurt to put this here
        if (useCudaLanczos)
            GenerateKernelLanczosWeights(params.sqrtSamples);
        else
            GenerateSampleDistribution();
    }
}

// delete and recreate buffer
static void ResizeImageBuffer() {
    params.bufferWidth  = params.imageWidth  * params.sqrtSamples;
    params.bufferHeight = params.imageHeight * params.sqrtSamples;
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &result_texture);
    if (offscreenBuffer) {
        cudaFree(offscreenBuffer);
        offscreenBuffer = nullptr;
    }

    InitBuffers();
}

static void WindowResizeCallback(int width, int height) {
    // set new OpenGL view port
    glViewport(0, 0, width, height);
    params.imageWidth  = width;
    params.imageHeight = height;

    ResizeImageBuffer();
}


static void WriteImageToDisk(const char* filename, bool writeLargeImage = false) {
    size_t bufferSize;
    uint32_t *cpuBuffer;
    int result;

    if (writeLargeImage) {
        // writes the entire image buffer to disk, possibly hundreds of MB!
        bufferSize = size_t(params.bufferWidth) * size_t(params.bufferHeight) * sizeof(*params.image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        if (useCudaLanczos) {
            cudaMemcpy(cpuBuffer, offscreenBuffer, bufferSize, cudaMemcpyDeviceToHost); // pull our result from the offscreen buffer
        }
        else {
            // pull our result from the PBO
            cudaGLMapBufferObject((void**) &params.image_buffer, pbo);
            cudaMemcpy(cpuBuffer, params.image_buffer, bufferSize, cudaMemcpyDeviceToHost);
            cudaGLUnmapBufferObject(pbo);
        }
        result = write_bmp(filename, params.bufferWidth, params.bufferHeight, (uint8_t*) cpuBuffer);
    }
    else {
        // writes the window contents to disk
        bufferSize = params.imageWidth * params.imageHeight * sizeof(*params.image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        glReadPixels(0, 0, params.imageWidth, params.imageHeight, GL_BGRA, GL_UNSIGNED_BYTE, cpuBuffer);
        result = write_bmp(filename, params.imageWidth, params.imageHeight, (uint8_t*) cpuBuffer);
    }

    free(cpuBuffer);
    if (result == 0) {
        printf("Wrote image to %s\n", filename);
    }
}

#define M_PI_F 3.14159265f
static float sinc(float x) {
    return sin(M_PI_F * x) / (M_PI_F * x);
}
// Lanczos sampling weights
static float Lanczos(float x, float y, float n) {
    float d = sqrtf(x*x + y*y);
    if (d == 0.0f)
        return 1.0f;
    else if (fabs(d) >= n)
        return 0.0f;
    else
        return sinc(d) * sinc(d / n);
}

// generates the sample distribution for image downscaling, including Lanczos weights
static void GenerateSampleDistribution() {
    uint32_t n = params.sqrtSamples;
    uint32_t spp = n * n;
    float fn = float(n);

    if (sampleDist) free(sampleDist);
    sampleDist = (float3*) malloc(spp * sizeof(*sampleDist));
    totalLanczosWeight = 0.0f;

    // stretch the kernel slightly depending on sample count. this improves image sharpness at medium to high spp.
    // do not touch without testing extensively, can have disastrous results!
    // sqrtf(n) * 0.5f improves quality at around 25 - 64 spp, with more samples image becomes too sharp
    float stretch = sqrtf(fn) * 0.5f;
    
    for (uint32_t y = 0; y < n; y++) {
        for (uint32_t x = 0; x < n; x++) {
            // sample distribution is a regular grid
            float u_adjust = ((x + 0.5f) / fn - 0.5f);
            float v_adjust = ((y + 0.5f) / fn - 0.5f);
            float weight = Lanczos(u_adjust * stretch, v_adjust * stretch, fn);
            sampleDist[y * n + x].x = u_adjust;
            sampleDist[y * n + x].y = v_adjust;
            sampleDist[y * n + x].z = weight;
            
            totalLanczosWeight += weight;
        }
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
    double scale = (2.0 / double(params.imageHeight)) * params.zoom;

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

     // toggle between CUDA and OpenGL/software Lanczos resampling
    case 'l':
        useCudaLanczos = !useCudaLanczos;
        ResizeImageBuffer();
        break;

    // toggle VSync
    case 'x':
        vsync = !vsync;
        SetVSync(vsync);
        break;

    // toggle text on screen
    case 'h':
        if (ctrlPressed) {
            hudState = !hudState;
        }
        else {
            helpState = !helpState;
        }
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

    case 'j': // decrease zoom speed
        zoomSpeed -= 0.01;
        zoomSpeed = fmax(zoomSpeed, 1.01);
        break;
    case 'k': // increase zoom speed
        zoomSpeed += 0.01;
        zoomSpeed = fmax(zoomSpeed, 1.01);
        break;

    case 'f': // decrease iterations
        params.iter = (uint32_t)(params.iter * 1.0 / zoomSpeed);
        if (params.iter == 0) params.iter = 1;
        break;
    case 'g': // increase iterations
        params.iter = (uint32_t) ceil(params.iter * zoomSpeed);
        break;

    case 'b': // decrease samples per pixel
        if (params.sqrtSamples > 1) {
            params.sqrtSamples--;
            ResizeImageBuffer();
        }
        break;

    case 'n': { // increase samples per pixel
        uint32_t newSpp = params.sqrtSamples + 1;
        if (((params.imageWidth  * newSpp) < (uint32_t) maxTextureSize) &&
            ((params.imageHeight * newSpp) < (uint32_t) maxTextureSize)) {

            params.sqrtSamples = newSpp;
            ResizeImageBuffer();
        }
    } break;

    case 'r': { // reset kernel params
        kernel_params newp = kernel_params();
        // do not change window size. if you really want to, call glutReshapeWindow(w,h) instead of ResizeImageBuffer().
        newp.imageWidth  = params.imageWidth;
        newp.imageHeight = params.imageHeight;
        params = newp;
        ResizeImageBuffer();
    } break;

    case 'i': { // write image and parameters to disk
        static char filename[64];
        static uint32_t prefix = uint32_t(GetTime() ^ (GetTime() >> 32));
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
            program_params dummy;
            kernel_params newp;
            if (ReadParamsFromDisk("input.par", &newp, &dummy)) {
                // do not change window size. if you really want to, call glutReshapeWindow(w,h) instead of ResizeImageBuffer().
                newp.imageWidth  = params.imageWidth;
                newp.imageHeight = params.imageHeight;
                params = newp;
                ResizeImageBuffer();
            }            
        }
        else {
            WriteParamsToDisk("input.par", params);
        }
    } break;
    }
}

static void SetVSync(int i) {
#if _WIN32
    if (WGLEW_EXT_swap_control)
        wglSwapIntervalEXT(i);
#else
    if (GLXEW_EXT_swap_control) {
        Display *dpy = glXGetCurrentDisplay();
        GLXDrawable drawable = glXGetCurrentDrawable();
        glXSwapIntervalEXT(dpy, drawable, i);
    }
#endif
    else
        printf("Missing extension to control vsync\n");
}

static void APIENTRY PrintDebugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam) {
    printf("%s\n", message);
#if _WIN32
    OutputDebugString(message);
    OutputDebugString("\n");
#endif
}

static void EnableGLDebugOutput() {
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
}

void GLWindowMain(int argc, char *argv[], const kernel_params& kp, uint32_t width, uint32_t height) {

    params = kp;
    params.imageWidth  = width;
    params.imageHeight = height;
    params.bufferWidth  = width  * params.sqrtSamples;
    params.bufferHeight = height * params.sqrtSamples;

    // create OpenGL context and window
#if _DEBUG
    glutInitContextFlags(GLUT_DEBUG);
#endif
    glutInit(&argc, (char**) argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA-Mandelbrot");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object GL_EXT_framebuffer_object ")) {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        return;
    }

#if _DEBUG
    EnableGLDebugOutput();
#endif

    SetVSync(vsync);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    InitBuffers();

    // register callbacks
    glutDisplayFunc(DisplayCallback);
    glutReshapeFunc(WindowResizeCallback);
    glutKeyboardFunc(KeyboardCallback);
    glutMouseWheelFunc(MouseWheelCallback);
    glutMouseFunc(MouseClickCallback);
    glutMotionFunc(MouseDragCallback);
   
    // start rendering main-loop
    glutMainLoop();
}

#endif
