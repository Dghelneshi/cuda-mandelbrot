/*
 *	calls bmp_output's print_bmp with the cuda generated data
 *
 */
#define WIDTH 1920
#define LENGTH 1080

#include <main.h>
#include <kernel.h>
#include <stdint.h>

static double maxlen = 1024.0;
static pos_t pos;

int bmpMain(int argc, const char*argc[]){
				//this goes into bmp_output
				char* imageData;
				launchKernel(imageData, WIDTH, HEIGHT, pos, maxlen);

				print_bmp ( WIDTH, HEIGHT, imageData );
}
