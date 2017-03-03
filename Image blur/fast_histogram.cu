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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
    return floorf(element * (float) numBins);
}


#include "fast_histogram_kernels.cu"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) {
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s elements bins", argv[0]);
        exit(1);
    }
    int elements = atoi(argv[1]);
    int bins = atoi(argv[2]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // adjust number of threads here
    unsigned int mem_size = sizeof(float) * elements;
    unsigned int bins_size = sizeof(int) * bins;

    // allocate host memory
    float * h_idata = (float *) malloc(mem_size);
    int * h_bins_zero = (int *) malloc(bins_size);
    int * h_bins_returned = (int *) malloc(bins_size);
    int * h_bins_correct = (int *) malloc(bins_size);
    // initialize the memory, compute correct answer
    for (unsigned int i = 0; i < bins; ++i) {
        h_bins_zero[i] = 0;
        h_bins_correct[i] = 0;
    }
    for (unsigned int i = 0; i < elements; ++i) {
        do {
            h_idata[i] = float(rand()) / float(RAND_MAX);
        } while(h_idata[i] >= 1.0f);
        h_bins_correct[whichBin(h_idata[i], bins)]++;
    }

    // allocate device memory
    float * d_idata;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_idata, mem_size));
    int * d_bins;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_bins, bins_size));
    // copy host memory to device
    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, mem_size,
                              cudaMemcpyHostToDevice));

    float tempTime, elapsedTime;
    elapsedTime = 1000000.0f;
    // run it 100 times, pick the best
    for (int i = 0; i < 100; i++) {
        // initialize bins to zero
        CUDA_SAFE_CALL(cudaMemcpy(d_bins, h_bins_zero, bins_size,
                                  cudaMemcpyHostToDevice));

        // TODO  WRITE THIS FUNCTION IN fast_histogram_kernels.cu
        setup(bins, elements);

        cudaEventRecord(start, 0);
        // TODO  WRITE THIS FUNCTION IN fast_histogram_kernels.cu
        histogram(bins, elements, d_bins, d_idata);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tempTime, start, stop);
        if (tempTime < elapsedTime) {
            elapsedTime = tempTime;
        }

        // TODO  WRITE THIS FUNCTION IN fast_histogram_kernels.cu
        finish();
    }

    CUDA_SAFE_CALL(cudaMemcpy(h_bins_returned, d_bins, bins_size,
                              cudaMemcpyDeviceToHost));

    printf( "Processing time: %f (ms)\n", elapsedTime);
    printf("Elements/s: %f\n", float(elements) * 1000.0f / elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check answer
    int insertedElements = 0;
    for (unsigned int i = 0; i < bins; ++i) {
        insertedElements += h_bins_returned[i];
        if (h_bins_returned[i] != h_bins_correct[i]) {
            printf("Error on bin %d, should be %d, is instead %d\n",
                   i, h_bins_correct[i], h_bins_returned[i]);
        }
    }
    if (insertedElements != elements) {
        printf("%d elements inserted (should be %d)\n",
               insertedElements, elements);
    }

    // clean up memory
    free(h_idata);
    free(h_bins_zero);
    free(h_bins_correct);
    free(h_bins_returned);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_bins));
}
