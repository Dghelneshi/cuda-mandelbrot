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

// TODO:
// this system isn't too bad to use, but error checking is not one of its strengths
// currently we only check arguments and whether any parameter name was wrong, but can't tell which one


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

// doesn't look very nice, but it's better than not checking at all
#define ERROR_CHECK(validArgc,p) (validArgc += (p) ? 2 : 0)

void ParseArgv(int argc, char** argv, kernel_params *kp, program_params *pp) {

    if (CheckParameter("-help",  argc, argv) ||
        CheckParameter("-?",     argc, argv) ||
        CheckParameter("--help", argc, argv)) {

        PrintCmdLineHelp();
        exit(0); // might be a bit harsh, but shouldn't happen during normal runtime
    }

    int validArgc = 1; // used for error checking

    // check for batch file first
    const char *batchFile = nullptr;
    ERROR_CHECK(validArgc, ReadParameter("-f", &batchFile, argc, argv));
    if (batchFile)
        ReadParamsFromDisk(batchFile, kp, pp);

    validArgc += (pp->useGl = (CheckParameter("-nogl", argc, argv) == 0));

    ERROR_CHECK(validArgc, ReadParameter("-o",      &pp->outputFile,    argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-ww",     &pp->window_height, argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-wh",     &pp->window_height, argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-w",      &kp->imageWidth,    argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-width",  &kp->imageWidth,    argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-h",      &kp->imageHeight,   argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-height", &kp->imageHeight,   argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-x",      &kp->centerX,       argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-y",      &kp->centerY,       argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-z",      &kp->zoom,          argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-b",      &kp->bailout,       argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-sx",     &kp->z0_x,          argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-sy",     &kp->z0_y,          argc, argv));
    ERROR_CHECK(validArgc, ReadParameter("-e",      &kp->exponent,      argc, argv));

    uint32_t i = 0;
    ERROR_CHECK(validArgc, ReadParameter("-i", &i, argc, argv));
    if (i > 0)
        kp->iter = i;

    cm_colors c = CM_COLORS_END;
    ERROR_CHECK(validArgc, ReadParameter("-c", &c, argc, argv));
    if ((c >= 0) && (c < CM_COLORS_END) && (c != CM_COLOR_DIST_END))
        kp->color = c;

    cm_type t = CM_FRACTAL_TYPES_END;
    ERROR_CHECK(validArgc, ReadParameter("-t", &t, argc, argv));
    if ((t >= 0) && (t < CM_FRACTAL_TYPES_END))
        kp->type = t;

    uint32_t spp = 0;
    ERROR_CHECK(validArgc, ReadParameter("-spp", &spp, argc, argv));
    if (spp != 0)
        kp->sqrtSamples = uint32_t(sqrtf(float(spp))+0.01f);

    kp->bufferWidth  = kp->imageWidth  * kp->sqrtSamples;
    kp->bufferHeight = kp->imageHeight * kp->sqrtSamples;

    // perform a rudimentary error check
    if (validArgc < argc) {
        printf("Warning: Unrecognized parameter.\n");
    }
}

// treats the given string as a command line and returns the parameters
// NOTE: modifies cmdline, does not allocate or copy strings
void ParseCmdLine(char* cmdline, kernel_params *kp, program_params *pp) {
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

    ParseArgv(static_argc, static_argv, kp, pp);
}

void PrintCmdLineHelp() {
    printf("\n" \
           "OPTIONS:\n" \
           "  -nogl\t\t\tUse command line interface only\n" \
           "  -x   <value>\t\tSet center x position\n" \
           "  -y   <value>\t\tSet center y position\n" \
           "  -z   <value>\t\tSet zoom level (z position, smaller means further in)\n" \
           "  -w   <value>\t\tSet image width  (no effect in -gl mode, see -ww)\n" \
           "  -h   <value>\t\tSet image height (no effect in -gl mode, see -wh)\n" \
           "  -i   <value>\t\tSet number of fractal iterations\n" \
           "  -b   <value>\t\tSet squared bailout radius for iteration\n" \
           "  -e   <value>\t\tSet exponent (fractal type 2 only)\n" \
           "  -c   <0-2,4>\t\tSet coloring method (3 is reserved)\n" \
           "  -t   <0-3>\t\tSet fractal type\n" \
           "  -sx  <value>\t\tSet starting z.x\n" \
           "  -sy  <value>\t\tSet starting z.y\n" \
           "  -spp <value>\t\tSet samples per pixel (Lanczos resampling), will be rounded down to next square number\n" \
           "  -o   <file>\t\tSet the output filename (default 'output.bmp', not used unless -nogl is specified)\n" \
           "  -ww  <value>\t\tSet window width  (with -gl)\n" \
           "  -wh  <value>\t\tSet window height (with -gl)\n" \
           "  -f   <file>\t\tProcess contents of <file> as command line (always executed first)\n" \
           "  \t\t\tCan be used to read in parameter files (.par) from GL window version\n");
}

// writes all kernel parameters to disk in the same format as a command line input
void WriteParamsToDisk(const char* filename, const kernel_params& params) {

    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open file '%s'.\n", filename);
        return;
    }

    uint32_t spp = params.sqrtSamples * params.sqrtSamples;

    fprintf(f, "-t %d -c %d -w %u -h %u -i %d -x %.17g -y %.17g -z %.17g -b %.17g -sx %.17g -sy %.17g -e %.17g -spp %u",
            params.type, params.color, params.imageWidth, params.imageHeight, params.iter, params.centerX, params.centerY, params.zoom,
            params.bailout, params.z0_x, params.z0_y, params.exponent, spp);

    fclose(f);
    printf("Wrote parameters to '%s'.\n", filename);
}

bool ReadParamsFromDisk(const char* filename, kernel_params *out_kp, program_params *out_pp) {

    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Could not open file '%s'.\n", filename);
        return false;
    }

    static char buffer[1024];
    size_t count = fread(buffer, 1, sizeof(buffer) - 1, f);
    buffer[count] = 0;

    // reset parameters
    kernel_params kp = kernel_params();
    program_params pp = program_params();

    ParseCmdLine(buffer, &kp, &pp);
    *out_kp = kp;
    *out_pp = pp;

    fclose(f);
    printf("Read parameters from '%s'.\n", filename);
    return true;
}
