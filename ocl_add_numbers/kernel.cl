#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "time_utils.h"
#include "ocl_utils.h"

#include <opencv2/imgcodecs/imgcodecs_c.h>

#define WIDTH (640 * 10)
#define HEIGHT (480 * 10)

#define MIN_REAL -2.0
#define MAX_REAL 1.0
#define MIN_IMAGINARY -1.1f
#define MAX_IMAGINARY (MIN_IMAGINARY + (MAX_REAL - MIN_REAL) *\
(HEIGHT) / (WIDTH))
//#define MAX_IMAGINARY 1.2
#define MAX_ITERATIONS 200

#define IMAGINARY_POS(y)\
(float)(MAX_IMAGINARY -\
(y) * ((MAX_IMAGINARY - MIN_IMAGINARY) / (float)((HEIGHT) - 1)))

#define REAL_POS(x)\
(float)(MIN_REAL + (x) * ((MAX_REAL - MIN_REAL) / (float)((WIDTH) - 1)))

__kernel void add_numbers(__global float *x_pos,
                          __global float *y_pos,
                          __global float *result)
{
    float real = REAL_POS(x_pos);
    float img = IMAGINARY_POS(y_pos);

    float z_real = real;
    float z_img = img;

    //bool is_inside = true;

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        float z_real_squared = z_real * z_real;
        float z_img_squared = z_img * z_img;

        if (z_real_squared + z_img_squared > 4)
        {
            // Not in the mandelbrot set
            result = ((float) i / (float)MAX_ITERATIONS) * 255.f;
        }
        float tmp = z_real_squared - z_img_squared + real;
        z_img = 2 * z_real * z_img + img;
        z_real = tmp;
    }

    result = 0.0;
}

