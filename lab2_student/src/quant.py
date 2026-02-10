import torch


def quantize_symmetric_int8(W: torch.Tensor):
    """Symmetric per-tensor int8 quantization.

    TODO(Quant-1): implement quantization
      - compute scale = max(abs(W))/127
      - compute integer Q = round(W/scale) clamped to [-127,127]
      - return Q as torch.int8 and scale as float

    Students should implement this function.
    """
    max_val = W.abs().max() # find the maximum absolute value in the tensor
    scale = max_val / 127 if max_val > 0 else 1.0

    Q = torch.round(W / scale).clamp(-127, 127).to(torch.int8) # quantize and clamp to int8 range
    return Q, scale

def dequantize_int8(Q: torch.Tensor, scale) -> torch.Tensor:
    """Dequantize int8 back to float16.

    TODO(Quant-2): implement dequantization
      - return (Q.float() * scale).half()
    """
    return (Q.float() * scale).half() # dequantize by multiplying with scale and convert float32 to float16
    
