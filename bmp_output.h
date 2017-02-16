#pragma once

// do not change. other formats are not implemented yet.
#define BYTESPERPIXEL 4

bool write_bmp(const char* filename, uint32_t width, uint32_t height, uint8_t *imageData);
