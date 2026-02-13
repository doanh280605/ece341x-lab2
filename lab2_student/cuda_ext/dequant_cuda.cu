#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Student TODO: implement this kernel
// Define a GPU kernel (runs on device, many threads in parallel)
__global__ void dequant_masked_kernel(const int8_t* __restrict__ q,
                                     const uint8_t* __restrict__ mask,
                                     half* __restrict__ out,
                                     float scale,
                                     int64_t n) {
  // TODO(CUDA-1): compute global index and dequantize masked values
  // int64_t i = ...;
  // if (i < n) { ... }
  // Students should implement this kernel.

  // Global linear index for this thread. 
  
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) { // bound check so extra threads do nothing
    // dequantize int8 -> float, then apply mask (0 => zero out), store as fp16
    float v = static_cast<float>(q[i]) * scale; // dequant, multiply by scale 
    out[i] = (mask[i] != 0) ? __float2half(v) : __float2half(0.0f); // apply mask 
    // keep value if mask = 1, otherwise, write zero 
    // convert to half fp16 for output
  }
}

torch::Tensor dequant_masked_int8_to_fp16_cuda(torch::Tensor qweight, double scale, torch::Tensor mask) {
  auto q = qweight.contiguous();
  auto m = mask.contiguous();

  const auto n = q.numel(); // total element to process
  auto out = torch::empty(q.sizes(), torch::TensorOptions().dtype(torch::kFloat16).device(q.device()));
  // allocate output tensor on same CUDA device. 

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);

  // TODO(CUDA-2): launch the kernel and check for errors
  // dequant_masked_kernel<<<blocks, threads>>>(...);
  // cudaError_t err = cudaGetLastError(); TORCH_CHECK(err==cudaSuccess, cudaGetErrorString(err));

  dequant_masked_kernel<<<blocks, threads>>>(
      reinterpret_cast<const int8_t*>(q.data_ptr<int8_t>()),
      reinterpret_cast<const uint8_t*>(m.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<float>(scale),
      n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "dequant_masked_kernel launch failed: ",
      cudaGetErrorString(err));

  return out;
}
