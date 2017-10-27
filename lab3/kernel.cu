/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void myHisto(unsigned int* input, unsigned int* bins, unsigned int num_elements, 
	unsigned int num_bins) {
	__shared__ unsigned int binL[4096]; // support num bins no more than 4096
	int step = 0;
	while (step < num_bins) {
		if (step + threadIdx.x < num_bins) binL[step + threadIdx.x] = 0;
		step += blockIdx.x;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < num_elements) {
		atomicAdd(binL[input[i]], 1);
		i += stride;
	}
	__syncthreads();

	step = 0;
	while (step < num_bins) {
		if (step + threadIdx.x < num_bins) atomicAdd(bins[step + threadIdx.x], binL[step + threadIdx.x]);
		step += blockIdx.x;
	}
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

	// Criteria for choosing gird size and block size
	// 1. shared memory size. 4k unsigned int use 16kB memory
	// 2. efficiency. If each block creates its own local bin, there will be #blocks * size_of_bins writes to global bin
	//	  Thus, the numbers a block deals must be levels larger than the size of bins. 

    // INSERT CODE HERE
    int grids = sqrt(num_elements) / 64;
    if (grids < 1) grids = 1;
    dim3 dimGrid(grids, 1, 1);
    dim3 dimBlock(512, 1, 1);
    myHisto <<<dimGrid, dimBlock>>> (input, bins, num_elements, num_bins);
}


