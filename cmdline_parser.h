#pragma once

#include "kernel.h"

void parseArgs(int argc, char** argv, kernel_params *p, bool *useGL, const char *outputFile);
void ParseCmdLine(char* cmdline, kernel_params *p, bool *useGL, const char *outputFile);
void WriteParamsToDisk(const char* filename, const kernel_params& params);
bool ReadParamsFromDisk(const char* filename, kernel_params* out_p);