/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // assume BLOCK_SIZE is power of 2
	__shared__ float res[2 * BLOCK_SIZE];
	int start = 2 * blockIdx.x * blockDim.x;
        if (start + threadIdx.x < size) 	
            res[threadIdx.x] = in[start + threadIdx.x];
        else res[threadIdx.x] = 0;
	if (BLOCK_SIZE + start + threadIdx.x < size)
            res[BLOCK_SIZE + threadIdx.x] = in[BLOCK_SIZE + start + threadIdx.x];
        else res[BLOCK_SIZE + threadIdx.x] = 0;

	for (int stride = BLOCK_SIZE; stride > 0; stride /= 2) {
		__syncthreads();
		if (threadIdx.x < stride) res[threadIdx.x] += res[threadIdx.x + stride];
	}
	if(threadIdx.x == 0) out[blockIdx.x] = res[0];
}
