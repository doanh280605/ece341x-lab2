#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Student TODO: implement this kernel
__global__ void dequant_masked_kernel(const int8_t* __restrict__ q,
                                     const uint8_t* __restrict__ mask,
                                     half* __restrict__ out,
                                     float scale,
                                     int64_t n) {
  // TODO(CUDA-1): compute global index and dequantize masked values
  // int64_t i = ...;
  // if (i < n) { ... }
  // Students should implement this kernel.
}

torch::Tensor dequant_masked_int8_to_fp16_cuda(torch::Tensor qweight, double scale, torch::Tensor mask) {
  auto q = qweight.contiguous();
  auto m = mask.contiguous();

  const auto n = q.numel();
  auto out = torch::empty(q.sizes(), torch::TensorOptions().dtype(torch::kFloat16).device(q.device()));

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  // TODO(CUDA-2): launch the kernel and check for errors
  // dequant_masked_kernel<<<blocks, threads>>>(...);
  // cudaError_t err = cudaGetLastError(); TORCH_CHECK(err==cudaSuccess, cudaGetErrorString(err));

  return out;
}
