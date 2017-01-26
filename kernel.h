#pragma once

#include <vector_types.h>
#include <stdint.h>
#include "main.h"

typedef struct pos_s {
    double centerX = -0.5;
    double centerY = 0.0;
    double zoom = 1.0;
} pos_t;

extern "C" {
    void launchKernel(uint32_t* image_buffer, uint32_t w, uint32_t h, pos_t pos, double maxlen2);
}
