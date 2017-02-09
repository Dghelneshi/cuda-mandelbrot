/*
 *	calls bmp_output's print_bmp with the cuda generated data
 *
 */
#define WIDTH 1920
#define HEIGHT 1080

#include "main.h"
#include "kernel.h"
#include "bmp_output.h"
#include <stdlib.h>
#include <stdint.h>

static double maxlen = 1024.0;
static pos_t pos;

int bmpMain(){
				//this goes into bmp_output
				uint32_t* memBuffer;	// image in memory
				uint32_t* gpuBuffer;	// image in gpu memory
				cudaMalloc;
				launchKernel(gpuBuffer, WIDTH, HEIGHT, pos, maxlen);
				malloc;
				cudaMemcpy 

				print_bmp ( WIDTH, HEIGHT, (char*)imageData );
				
				return EXIT_SUCCESS;
}
