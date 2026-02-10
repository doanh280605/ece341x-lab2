import torch


def quantize_symmetric_int8(W: torch.Tensor):
    """Symmetric per-tensor int8 quantization.

    TODO(Quant-1): implement quantization
      - compute scale = max(abs(W))/127
      - compute integer Q = round(W/scale) clamped to [-127,127]
      - return Q as torch.int8 and scale as float

    Students should implement this function.
    """
    raise NotImplementedError("Implement quantize_symmetric_int8 in student version")


def dequantize_int8(Q: torch.Tensor, scale) -> torch.Tensor:
    """Dequantize int8 back to float16.

    TODO(Quant-2): implement dequantization
      - return (Q.float() * scale).half()
    """
    raise NotImplementedError("Implement dequantize_int8 in student version")
