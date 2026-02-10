import time
import torch

@torch.no_grad()
def accuracy(model, loader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def model_size_bytes_float_params(model) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total

@torch.no_grad()
def time_inference(model, x, iters=50, warmup=10, device="cuda"):
    model.eval()
    x = x.to(device)
    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters
