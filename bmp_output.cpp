
/*
 *	Responsible for writing a buffer of RGB data to stdout
 *	If you want it in a file, then redirect stdout.
 */

#include <stdlib.h>
#include <stdio.h>

#define BYTE_ALIGNMENT 4  /* probably shouldn't change this */
#define BYTE_OFFSET 54
#define HEADER_SIZE 40

/*
 * thingy that every bmp file needs at the beginning of it.
 */
#pragma pack(1) /* pack the struct for easy printing */
struct BitmapHeader
{
	char	fileType[2];				/* 'b', 'm' for bitmap*/
	int		fileSize;						/* size of the whole file in bytes */
	int		fileReserved;				/* 0 */
	int		byteOffset;					/* byte offset from file start to image data */
	int		infoHeaderSize;			/* 40 */
	int		imageWidth;					/* width  in px*/
	int	 	imageHeight;				/* height in px */
	short	imagePlanes;				/* 1 on non-alien devices */
	short	bitsPerPixel;				/* 24 for RGB, 32 for RGBA */
	int		compressionType;		/* 0 for no compression */
	int		imageSize;					/* size of compressed image (0 for uncompressed)*/
	int		pixelsPerMeterX;		/*   let's set these two to 0 and hope */
	int		pixelsPerMeterY;		/*   nothing bad will happen           */
	int 	colorTableSize;			/* number of colors in colortable 0 means no table*/
	int		importantColorSize;	/* if 0, then all colors are 'important' */
};


/*
 * prints a .bmp image to stdout where:
 * width is the image width in pixels,
 * height is the image height in pixels
 * and imageData is the RGB information of the pixels.
 */
int print_bmp( int width, int height, char *imageData  )
{
	struct BitmapHeader bmpHeader;

	/* to assure alignment of "lines" along 4byte borders 
	 * this variable contains the exact amount of bytes a line should have
	 **/
	int lineSize = width +1;
	lineSize *= BYTES_PER_PIXEL;
	lineSize /= BYTE_ALIGNMENT;
	lineSize *= BYTE_ALIGNMENT;
	
	/* write some actual data into the header */
	bmpHeader.fileType[0] = 'B';
	bmpHeader.fileType[1] = 'M';
	bmpHeader.byteOffset = BYTE_OFFSET;
	bmpHeader.fileReserved = 0;
	bmpHeader.infoHeaderSize = HEADER_SIZE;
	bmpHeader.imageWidth = width;
	bmpHeader.imageHeight = height;
	bmpHeader.imagePlanes = 1;
	bmpHeader.bitsPerPixel = 8 * BYTES_PER_PIXEL;
	bmpHeader.compressionType = 0; /* uncompressed */
	bmpHeader.imageSize = (lineSize * height);
	bmpHeader.fileSize = bmpHeader.byteOffset + bmpHeader.imageSize;
	bmpHeader.pixelsPerMeterX = 0; /* this doesn't apply to our use-case */
	bmpHeader.pixelsPerMeterY = 0;
	bmpHeader.colorTableSize = 0;	/* no color table */
	bmpHeader.importantColorSize = 0; /* no particularly important colors */

	/* print the header  */
	fwrite( 
									&bmpHeader, 
									sizeof(char), 
									(size_t)HEADER_SIZE, 
									stdout );	

	/* print the padding between header and data */
	fwrite(
									imageData, /* or literally anything else! */
									sizeof(char),
									(size_t) (BYTE_OFFSET - HEADER_SIZE), /* size of padding */
								 	stdout );	

	/* print the image */
	fwrite( 
									imageData, 
									sizeof(char), 
									(size_t)(BYTES_PER_PIXEL * bmpHeader.imageSize),
									stdout );



	return EXIT_SUCCESS;
}



