import os, sys
import torch
sys.path.append(os.path.dirname(__file__))

def ref(q, scale, mask):
    return (q.float() * scale).half() * mask.to(torch.float16)

def main():
    import dequant_ext
    device = "cuda"
    torch.manual_seed(0)
    n = 2_000_000
    q = torch.randint(-127, 128, (n,), device=device, dtype=torch.int8)
    mask = (torch.rand(n, device=device) > 0.3).to(torch.uint8)
    scale = 0.03
    out = dequant_ext.dequant_masked_int8_to_fp16(q, float(scale), mask)
    r = ref(q, scale, mask)
    max_err = (out - r).abs().max().item()
    print("max_abs_err:", max_err)
    assert max_err < 1e-2
    mv = out[mask == 0]
    if mv.numel() > 0:
        assert mv.abs().max().item() == 0.0
    print("Correctness OK.")

if __name__ == "__main__":
    main()
