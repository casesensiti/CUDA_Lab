/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    int tileL = blockDim.x;
	__shared__ float dA[TILE_SIZE][TILE_SIZE]; // a tile in A
	__shared__ float dB[TILE_SIZE][TILE_SIZE]; // a tile in B

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float val = 0;

	for (int p = 0; p < k; p += tileL) {
		// load dA
		if (row < m && (p + threadIdx.x) < k) dA[threadIdx.y][threadIdx.x] = A[row * k + p + threadIdx.x];
		else dA[threadIdx.y][threadIdx.x] = 0;
		// load dB
		if (col < n && (p + threadIdx.y) < k) dB[threadIdx.y][threadIdx.x] = B[(p + threadIdx.y) * n + col];
		else dB[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();
		// compute 
		for (int i = 0; i < tileL; i++) val += dA[threadIdx.y][i] * dB[i][threadIdx.x];

		__syncthreads();
	}
	if (row < m && col < n) C[row * n + col] = val;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dimGrid((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm <<<dimGrid, dimBlock>>> (m, n, k, A, B, C);

}


