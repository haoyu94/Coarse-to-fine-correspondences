#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cstdio>
#include <cuda_runtime_api.h>

#include "grouping_gpu.h"
#include "cuda_utils.h"

__global__ void grouping_kernel(int b, int n, int m, int nsample, const int *__restrict__ xyz, int *__restrict__ idx) {
    // xyz: (B, N, 1)
    // output:
    //      idx: (B, M, nsample)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    xyz += bs_idx * n; // query begining address
    idx += bs_idx * m * nsample + pt_idx * nsample; // idx beginning address
    int query_n = pt_idx % m; // query_n: the query number [0, n-1]

    // printf("bs_idx: %d, query_n: %d, pt_idx: %d n: %d, m: %d\n", bs_idx, query_n, pt_idx, n, m);

    curandState state;
    curand_init(clock64(), pt_idx, 0, &state);

    int cnt = 0;
    int sit = 0;
    for( int k = 0; k < n; ++k){
        int node_n = xyz[k];
        if( node_n == query_n ){
            sit = cnt;
            if( cnt < nsample){
                ++cnt;
            } else {
                sit = curand_uniform(&state) * nsample;
            }
            idx[sit] = k;
        }
    }
    // printf("cnt: %d\n",cnt);
    // for(int i=0; i<5;++i){
    //    printf("cnt: %d, rand: %f\n", cnt, curand_uniform(&state));
    // }
    //for(int k=cnt; k<nsample;++k){
    //    int sit = curand_uniform(&state) * cnt;
        // printf("k: %d, sit: %d\n",cnt,sit);
    //    idx[k] = idx[sit];
    //}
}

void grouping_kernel_launcher(int b, int n, int m, int nsample, const int *xyz, int *idx, cudaStream_t stream) {
    // xyz: (B, N, 1)
    // output:
    //      idx: (B, M, nsample)
    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    grouping_kernel<<<blocks, threads, 0, stream>>>(b, n, m, nsample, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
