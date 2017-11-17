/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d0, *B_d0, *C_d0;
    float *A_d1, *B_d1, *C_d1;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;

    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000000;
    } else if (argc == 2) {
        VecSize = atoi(argv[1]);   
    }
    else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }
    unsigned SeqSize = 500000;
    if (SeqSize > VecSize) SeqSize = VecSize;
   
    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMalloc((void**) &A_d0, sizeof(float)*SeqSize); 
    cudaMalloc((void**) &B_d0, sizeof(float)*SeqSize); 
    cudaMalloc((void**) &C_d0, sizeof(float)*SeqSize); 

    cudaMalloc((void**) &A_d1, sizeof(float)*SeqSize); 
    cudaMalloc((void**) &B_d1, sizeof(float)*SeqSize); 
    cudaMalloc((void**) &C_d1, sizeof(float)*SeqSize); 

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    //  Multi-stream
    printf("Launching multi-stream processing..."); fflush(stdout);
    startTime(&timer);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    for (unsigned i = 0; i < VecSize; i += SeqSize * 2) {
        int size0 = SeqSize, size1 = SeqSize;
        if (i + SeqSize > VecSize) {
            size0 = VecSize - i;
            size1 = 0;
        } else if (i + 2 * SeqSize > VecSize) size1 = VecSize - i -  SeqSize;

        cudaMemcpyAsync(A_d0, A_h + i, size0 * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(B_d0, B_h + i, size0 * sizeof(float), cudaMemcpyHostToDevice, stream0);

        cudaMemcpyAsync(A_d1, A_h + i + size0, size1 * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(B_d1, B_h + i + size0, size1 * sizeof(float), cudaMemcpyHostToDevice, stream1);

        VecAdd<<<(size0 - 1) / 256 + 1, 256, 0, stream0>>> (size0, A_d0, B_d0, C_d0);
        VecAdd<<<(size1 - 1) / 256 + 1, 256, 0, stream1>>> (size1, A_d1, B_d1, C_d1);

        cudaMemcpyAsync(C_h + i, C_d0, size0 * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(C_h + i + size0, C_d1, size1 * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



/*
    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicVecAdd(A_d, B_d, C_d, VecSize); //In kernel.cu

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
*/
    
    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    return 0;

}
