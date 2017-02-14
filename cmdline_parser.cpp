#if _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define snprintf sprintf_s
#define fprintf fprintf_s
#define fscanf fscanf_s
#endif

#include <cstdio>
#include <math.h>
#include <cstring>
#include <cstdlib>

#include "kernel.h"
#include "cmdline_parser.h"


#define MAX_ARGV 128
static char *static_argv[MAX_ARGV] = { 0 };
static int static_argc = 1;


// reads a value from a string argument
template<typename T>
T Read(const char *argv);

template<> double Read(const char *argv) {
    return strtod(argv, nullptr);
}
template<> uint32_t Read(const char *argv) {
    return strtoul(argv, nullptr, 0);
}
template<> cm_type Read(const char *argv) {
    return (cm_type) strtoul(argv, nullptr, 0);
}
template<> cm_colors Read(const char *argv) {
    return (cm_colors) strtoul(argv, nullptr, 0);
}
template<> const char* Read(const char *argv) {
    return argv;
}

// tries to read a parameter from the command line
template<typename T>
static int ReadParameter(const char *parameter, T *res_p, int argc = static_argc, char *argv[] = static_argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(parameter, argv[i]) == 0 && (i + 1 < argc)) {
            *res_p = Read<T>(argv[i + 1]);
            return i;
        }
    }
    return 0;
}

static int CheckParameter(const char *parameter, int argc = static_argc, char *argv[] = static_argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(parameter, argv[i]) == 0) {
            return i;
        }
    }
    return 0;
}

void parseArgs(int argc, char** argv, kernel_params *p, bool *useGL = nullptr, const char *outputFile = nullptr) {

    const char *batchFile = nullptr;
    ReadParameter("-f", &batchFile, argc, argv);
    if (batchFile)
        ReadParamsFromDisk(batchFile, p);

    if (useGL)
        *useGL = CheckParameter("-gl", argc, argv) != 0;

    if (outputFile)
        ReadParameter("-o", &outputFile, argc, argv);

    ReadParameter("-w", &p->width, argc, argv);
    ReadParameter("-h", &p->height, argc, argv);
    ReadParameter("-i", &p->iter, argc, argv);
    ReadParameter("-x", &p->centerX, argc, argv);
    ReadParameter("-y", &p->centerY, argc, argv);
    ReadParameter("-z", &p->zoom, argc, argv);
    ReadParameter("-b", &p->bailout, argc, argv);
    ReadParameter("-sx", &p->z0_x, argc, argv);
    ReadParameter("-sy", &p->z0_y, argc, argv);
    ReadParameter("-e", &p->exponent, argc, argv);
    ReadParameter("-c", &p->color, argc, argv);
    ReadParameter("-t", &p->type, argc, argv);

    uint32_t spp = 1;
    if (ReadParameter("-spp", &spp, argc, argv) != 0) {
        p->sqrtSamples = uint32_t(sqrtf(float(spp)) + 0.5f);
    }
}

// treats the given string as a command line and returns the parameters
// NOTE: modifies cmdline, does not allocate or copy strings
void ParseCmdLine(char* cmdline, kernel_params *p, bool *useGL = nullptr, const char *outputFile = nullptr) {
    static_argc = 1;

    while (*cmdline && (static_argc < MAX_ARGV)) {

        // skip control characters and space
        while (*cmdline && ((*cmdline <= 32) || (*cmdline > 126))) {
            cmdline++;
        }

        if (*cmdline) {
            // set current argument pointer
            static_argv[static_argc] = cmdline;
            static_argc++;

            // skip to end of current argument
            while (*cmdline && ((*cmdline > 32) && (*cmdline <= 126))) {
                cmdline++;
            }

            // terminate the string of current argument
            if (*cmdline) {
                *cmdline = 0;
                cmdline++;
            }
        }
    }

    parseArgs(static_argc, static_argv, p, useGL, outputFile);
}

// writes all parameters to disk in the same format as a command line input
void WriteParamsToDisk(const char* filename, const kernel_params& params) {

    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }

    uint32_t width = params.width / params.sqrtSamples;
    uint32_t height = params.height / params.sqrtSamples;
    uint32_t spp = params.sqrtSamples * params.sqrtSamples;

    fprintf(f, "-t %d -c %d -w %u -h %u -i %d -x %.17g -y %.17g -z %.17g -b %.17g -sx %.17g -sy %.17g -e %.17g -spp %u",
            params.type, params.color, width, height, params.iter, params.centerX, params.centerY, params.zoom,
            params.bailout, params.z0_x, params.z0_y, params.exponent, spp);

    fclose(f);
    printf("Wrote parameters to %s\n", filename);
}

bool ReadParamsFromDisk(const char* filename, kernel_params* out_p) {

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Could not open file %s\n", filename);
        return false;
    }

    static char buffer[1024];
    size_t count = fread(buffer, 1, sizeof(buffer) - 1, f);
    buffer[count] = 0;

    kernel_params p = kernel_params(); // set standard values

    ParseCmdLine(buffer, &p);
    *out_p = p;

    fclose(f);
    printf("Read parameters from %s\n", filename);
    return true;
}
