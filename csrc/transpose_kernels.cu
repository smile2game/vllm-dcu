#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


namespace vllm {
template <typename T>
__global__ void trans_w16_gemm_cudakernel(int64_t num_kernels,T* dst,const T* src,int64_t row,int64_t col)
{
    int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_kernels) return;

    int64_t j=id%row; 
    int64_t i=id/row;

    dst[i*row+j]=src[j*col+i];
}


void trans_w16_gemm_cuda(half* dst,const half* src,int64_t row,int64_t col){
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t num_kernels=row*col;
  int block_size=256;
  trans_w16_gemm_cudakernel<<<(num_kernels+block_size-1)/block_size,block_size, 0, stream>>>(num_kernels,dst,src,row,col);
}
}   // namespace vllm

void trans_w16_gemm(torch::Tensor dst,torch::Tensor src,int64_t row,int64_t col){
  const at::cuda::OptionalCUDAGuard device_guard(device_of(src));
  vllm::trans_w16_gemm_cuda(
              (half*)dst.data_ptr(),
              (const half*)src.data_ptr(),
              row,
              col
            );
}