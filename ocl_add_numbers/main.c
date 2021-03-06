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

cl_mem create_result_buffer(void)
{
    cl_int error;
	float *host = malloc(sizeof(cl_float));

    cl_mem dev_vec = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*WIDTH*HEIGHT, host, &error);
    ocl_err(error);
	free(host);
    return dev_vec;
}

void render_mandelbrot(CvMat * output_image)
{
	cl_int error;
    // Create device buffers.
    cl_mem dev_result = create_result_buffer();
    cl_float *host_result = malloc(sizeof(cl_float));

	// Create kernel
    cl_kernel kernel = clCreateKernel(g_program, "add_numbers", &error);
    ocl_err(error);

	// Set kernel arguments
    int arg_num = 0;
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &x_pos));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &y_pos));
    ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_result));

    for (int pos_y = 0; pos_y < output_image->rows; ++pos_y)
    {
        for (int pos_x = 0; pos_x < output_image->cols; ++pos_x)
        {
            //float result = calc_mandel_pixel(pos_x, pos_y);
            //output_image->data.fl[pos_y * WIDTH + pos_x] = result;
        }
    }
}

int main(int argc, char *argv[])
{
	CvMat * output_image = cvCreateMat(HEIGHT, WIDTH, CV_32FC1);

    cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
    create_program("kernel.cl", "");

    time_measure_start("total");
	render_mandelbrot(output_image);
    time_measure_stop_and_print("total");

	cvSaveImage("mandelbrot.png", output_image, 0);
}
