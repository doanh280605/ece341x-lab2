import argparse, os, json
import torch

from .data import make_loaders
from .models import get_model, iter_prunable_params
from .utils_measure import accuracy
from .prune import make_masks_for_model, apply_mask_
from .quant import quantize_symmetric_int8, dequantize_int8

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/vgg_cifar10.pt")
    ap.add_argument("--variant", type=str, default="vgg16_bn")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sparsity", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    _, test_loader = make_loaders(128)
    device = args.device

    model = get_model(args.variant).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    # Robust checkpoint loading: support wrapped checkpoints with 'state_dict' or 'model_state_dict',
    # and strip common prefixes like 'module.' and 'backbone.'
    loaded = torch.load(args.ckpt, map_location="cpu")
    if isinstance(loaded, dict):
        if 'state_dict' in loaded:
            state_dict = loaded['state_dict']
        elif 'model_state_dict' in loaded:
            state_dict = loaded['model_state_dict']
        else:
            state_dict = loaded
    else:
        state_dict = loaded

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            if nk.startswith('backbone.'):
                nk = nk[len('backbone.'):]
            new_sd[nk] = v
        state_dict = new_sd

    # Load non-strictly and warn about missing/unexpected keys
    try:
        res = model.load_state_dict(state_dict, strict=False)
        if getattr(res, 'missing_keys', None):
            print(f"Warning: missing keys when loading checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
        if getattr(res, 'unexpected_keys', None):
            print(f"Warning: unexpected keys in checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
    except Exception:
        # Fallback: filter keys to those matching the model
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        res = model.load_state_dict(filtered, strict=False)
        print("Loaded filtered checkpoint (only keys matching model).")
        if getattr(res, 'missing_keys', None):
            print(f"Warning: missing keys when loading filtered checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
        if getattr(res, 'unexpected_keys', None):
            print(f"Warning: unexpected keys in filtered checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
    model.eval()

    named_weights = list(iter_prunable_params(model))
    masks = make_masks_for_model(named_weights, sparsity=args.sparsity)

    # Apply pruning + quant to all weights, then swap in dequantized fp16 weights (python path)
    scales = {}
    for name, W in named_weights:
        m = masks[name]
        W_eff = W * m.to(dtype=W.dtype)
        Q, s = quantize_symmetric_int8(W_eff)
        W_hat = dequantize_int8(Q, s) * m.to(torch.float16)
        W.data = W_hat.to(device).to(dtype=W.dtype)
        scales[name] = float(s)

    acc = accuracy(model, test_loader, device=device)
    out = {"sparsity": float(args.sparsity), "test_acc": float(acc), "num_layers": len(named_weights), "scales": scales}
    with open("results/quant_all_export.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/quant_all_export.json")

if __name__ == "__main__":
    main()
