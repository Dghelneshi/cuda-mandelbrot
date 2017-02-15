#pragma once

#include "kernel.h"

struct program_params {
    bool useGl;
    const char* outputFile;
    uint32_t window_width;
    uint32_t window_height;

    program_params(): useGl(false), outputFile(nullptr), window_width(1280), window_height(720) { }
};

void ParseArgv(int argc, char** argv, kernel_params *p, program_params *pp);
void ParseCmdLine(char* cmdline, kernel_params *p, program_params *pp);
void WriteParamsToDisk(const char* filename, const kernel_params& params);
bool ReadParamsFromDisk(const char* filename, kernel_params* out_p, program_params *out_pp);
void PrintCmdLineHelp();