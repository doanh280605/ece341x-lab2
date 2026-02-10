#include <torch/extension.h>

torch::Tensor dequant_masked_int8_to_fp16_cuda(torch::Tensor qweight, double scale, torch::Tensor mask);

torch::Tensor dequant_masked_int8_to_fp16(torch::Tensor qweight, double scale, torch::Tensor mask) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
  TORCH_CHECK(qweight.scalar_type() == at::kChar, "qweight must be torch.int8");
  TORCH_CHECK(mask.scalar_type() == at::kByte, "mask must be torch.uint8");
  TORCH_CHECK(qweight.numel() == mask.numel(), "qweight and mask must have same numel");
  return dequant_masked_int8_to_fp16_cuda(qweight, scale, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_masked_int8_to_fp16", &dequant_masked_int8_to_fp16,
        "Masked int8 -> fp16 dequant (CUDA)");
}
