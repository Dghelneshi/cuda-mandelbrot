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
#include <cuda_runtime.h>

#define BUFSIZE (WIDTH * HEIGHT * BYTESPERPIXEL)

static double maxlen = 1024.0;
static pos_t pos;

int bmpMain(){
				//this goes into bmp_output
				uint32_t* memBuffer;				// image in memory
				uint32_t* gpuBuffer;				// image in gpu memory
				cudaMalloc( (void**)&gpuBuffer, BUFSIZE);	// allocate gpu buffer
				launchKernel(gpuBuffer, WIDTH, HEIGHT, pos, maxlen);
				memBuffer = (uint32_t*)malloc(BUFSIZE);		// allocate memory buffer
				cudaMemcpy(
						memBuffer,
						gpuBuffer,
						BUFSIZE,
						cudaMemcpyDeviceToHost);	// copy the image from gpu into memory

				print_bmp ( WIDTH, HEIGHT, (char*)memBuffer );
				
				return EXIT_SUCCESS;
}
