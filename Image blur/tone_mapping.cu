/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#undef TINYEXR_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

// includes, project
#include <cuda.h>
#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)


// note: __host__ __device__ means "compile this for both CPU and GPU"
// since we're using it on both. Cool trick, huh?
// assumes "element" is [0.0f-1.0f)
__host__ __device__ int whichBin(float element, int numBins) {
    int bin = floorf(element * (float) numBins);
    if(bin >= numBins)
      return numBins-1;
    return bin;
}


#include "tone_mapping_kernels.cu"


void normalizeCDF(unsigned int* cdf, float* cdf_norm, int n) {
    const float normConstant = 1.f / cdf[n-1];

    for(int i=0; i<n; ++i) {
        cdf_norm[i] = cdf[i] * normConstant;
    }
}

uint8_t* tonemap(float* hdrImage, int width, int height, int numBins,
                 float* cdf_norm, float minLogY, float maxLogY)
{
    uint8_t* ret = (uint8_t*) malloc(3*width*height*sizeof(uint8_t));

    float logYRange = maxLogY - minLogY;

    for(int i=0; i<width*height*3; i+=3) {
        float r = hdrImage[i];
        float g = hdrImage[i+1];
        float b = hdrImage[i+2];

        float x, y, logY;
        rgb_to_xyY(r, g, b, &x, &y, &logY);
        int binIdx = min( numBins - 1, int((numBins * (logY - minLogY)) / logYRange));

        float yNew = cdf_norm[binIdx];
        float xNew = x * (yNew / y);
        float zNew = (1-x-y) * (yNew / y);

        float rNew = ( xNew *  3.2406f ) + ( yNew * -1.5372f ) + ( zNew * -0.4986f );
        float gNew = ( xNew * -0.9689f ) + ( yNew *  1.8758f ) + ( zNew *  0.0415f );
        float bNew = ( xNew *  0.0557f ) + ( yNew * -0.2040f ) + ( zNew *  1.0570f );        


        ret[i]   = max(0.f, min(255.f, 255.f * rNew));
        ret[i+1] = max(0.f, min(255.f, 255.f * gNew));
        ret[i+2] = max(0.f, min(255.f, 255.f * bNew));
    }

    return ret;
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    runTest( argc, argv);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) {

    int numBins = 1024;
    const char* outfilename = "equalizedImage.png";

    if (argc != 2) {
        fprintf(stderr, "Usage: %s filename", argv[0]);
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    float* imageRGBA;
    int width = 0;
    int height = 0;
    const char* err;
    int ret = LoadEXR(&imageRGBA, &width, &height, argv[1], &err);

    if(ret != 0) {
        fprintf(stderr, "Unable to open file: %s\n", argv[1]);
        exit(1);
    }

    float* image = (float*) malloc(3*width*height*sizeof(float));
    for(int i=0; i<width*height; ++i) {
        image[i*3] = imageRGBA[i*4];
        image[i*3+1] = imageRGBA[i*4+1];
        image[i*3+2] = imageRGBA[i*4+2];
    }

    unsigned int *h_cdf = (unsigned int*) malloc(numBins*sizeof(unsigned int));
    for(int i=0; i<numBins; ++i) {
        h_cdf[i] = 0;
    }

    unsigned int* d_cdf;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_cdf, numBins*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_cdf, h_cdf, numBins*sizeof(unsigned int), cudaMemcpyHostToDevice));

    float minLogY = FLT_MAX;
    float maxLogY = -FLT_MAX;

    cudaEventRecord(start, 0);
    computeHistogramAndCDF(image, width, height, numBins, d_cdf, minLogY, maxLogY);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Your code executed in %f ms\n", elapsedTime);

    CUDA_SAFE_CALL(cudaMemcpy(h_cdf, d_cdf, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    float* cdf_norm = (float*) malloc(numBins*sizeof(float));
    normalizeCDF(h_cdf, cdf_norm, numBins);

    uint8_t* output = tonemap(image, width, height, numBins, cdf_norm, minLogY, maxLogY);

    stbi_write_png(outfilename, width, height, 3, output, 3*width*sizeof(uint8_t));
    printf("Output image written to %s\n", outfilename);
}
