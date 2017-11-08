/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE
__global__ void partialScan(float *out, float *in, float *out_d, unsigned in_size) {
	__shared__ buf[BLOCK_SIZE * 2];
	int tx = threadIdx.x, offset = blockIdx.x * blockDim.x * 2;
	// note: should buf set to 0 if out of range?
	if (tx + offset < in_size) buf[tx] = in[tx + offset];
	if (tx + offset + BLOCK_SIZE < in_size) buf[tx + BLOCK_SIZE] = in[tx + offset + BLOCK_SIZE];

	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int i = (tx + 1) * stride * 2 - 1;
		if (i < BLOCK_SIZE * 2) buf[i] += buf[i - stride];
	}
	for (int stride = BLOCK_SIZE/2; stride >= 1; stride /= 2) {
		__syncthreads();
		int i = (tx + 1) * stride * 2 - 1;
		if (i + stride < BLOCK_SIZE * 2) buf[i + stride] += buf[i];
	}
	__syncthreads();
	// copy to out_d
	if (!tx) {
		if (blockIdx.x == blockDim.x - 1) out_d[0] = 0;
		else out_d[blockIdx.x + 1] = buf[BLOCK_SIZE * 2 - 1];
	}
	// copy to out
	if (tx + offset < in_size) out[tx + offset] = buf[tx];
	if (tx + offset + BLOCK_SIZE < in_size) out[tx + offset + BLOCK_SIZE] = buf[tx + BLOCK_SIZE];
}

__global__ void addVec(float *out, float *toAdd, unsigned in_size) {
	int tx = threadIdx.x, offset = blockIdx.x * blockDim.x * 2;
	if (tx + offset < in_size) out[tx + offset] += toAdd[blockIdx.x];
	if (tx + offset + BLOCK_SIZE < in_size) out[tx + offset + BLOCK_SIZE] = toAdd[blockIdx.x];
}



/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
    // INSERT CODE HERE
    cudaError_t cuda_ret;
    float *out_b;
    int gridLen = (in_size - 1) / (2 * BLOCK_SIZE) + 1;
	dim3 gridDim(gridLen, 1, 1);
	dim3 blockDim(BLOCK_SIZE, 1, 1);
	// allocate space for last value in first n - 1 block
	cuda_ret = cudaMalloc((void**)&out_b, gridLen * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    // launch kernel
    partialScan <<<grodDim, blockDim>>> (out, in, out_d, in_size);

    // preScan and plus back out_d if needed
    if (gridLen > 1) {
    	float *out_bscaned;
    	cuda_ret = cudaMalloc((void**)&out_bscaned, gridLen * sizeof(float));
    	if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    	preScan(out_bscaned, out_b, gridLen);
    	addVec <<<grodDim, blockDim>>> (out, out_bscaned, in_size);
    	cudaFree(out_bscaned);
    }
    cudaFree(out_b);
}

