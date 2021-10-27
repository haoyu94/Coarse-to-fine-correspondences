#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int grouping_wrapper(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void grouping_kernel_launcher(int b, int n, int m, int nsample, const int *xyz, int *idx, cudaStream_t stream);

#endif
