#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdint.h>
#include "bmp_output.h"

#define BMP_FILE_HEADER_SIZE (14) // size of bitmap file header

#pragma pack(push,1) /* no padding between struct members */
struct BitmapHeader {
    // bitmap file header
        char fileType[2];        /* 'B', 'M' for bitmap*/
    uint32_t fileSize;           /* size of the whole file in bytes */
    uint32_t reserved;           /* 0 */
    uint32_t byteOffset;         /* byte offset from file start to image data */
    // DIB header
    uint32_t infoHeaderSize;     /* size of the DIB header */
     int32_t imageWidth;
     int32_t imageHeight;        /* image is top-down if this is negative */
    uint16_t imagePlanes;        /* 1 on non-alien devices */
    uint16_t bitsPerPixel;       /* 24 for RGB, 32 for XRGB */
    uint32_t compressionType;    /* 0 for no compression */
    uint32_t imageSize;          /* size of compressed image (0 for uncompressed)*/
     int32_t pixelsPerMeterX;
     int32_t pixelsPerMeterY;
    uint32_t colorTableSize;     /* number of colors in colortable 0 means no table*/
    uint32_t importantColorSize; /* if 0, then all colors are 'important' */
};
#pragma pack(pop)


int write_bmp(const char* filename, uint32_t width, uint32_t height, uint8_t* imageData) {

    uint32_t imageSize = width * height * BYTESPERPIXEL;

    struct BitmapHeader bmpHeader = {0};
    bmpHeader.fileType[0]     = 'B';
    bmpHeader.fileType[1]     = 'M';
    bmpHeader.byteOffset      = sizeof(BitmapHeader);
    bmpHeader.fileSize        = bmpHeader.byteOffset + imageSize;
    bmpHeader.infoHeaderSize  = sizeof(BitmapHeader) - BMP_FILE_HEADER_SIZE;
    bmpHeader.imageWidth      = width;
    bmpHeader.imageHeight     = height;
    bmpHeader.imagePlanes     = 1;
    bmpHeader.bitsPerPixel    = 8 * BYTESPERPIXEL;
    
    FILE* outputFile = fopen(filename, "wb");
    if (!outputFile) return 1;

    /* write the header  */
    fwrite(&bmpHeader, 1, sizeof(bmpHeader), outputFile);
    /* write the image (no padding per row, assumes BYTESPERPIXEL == 4) */
    fwrite(imageData, 1, imageSize, outputFile);

    fclose(outputFile);
    return 0;
}
