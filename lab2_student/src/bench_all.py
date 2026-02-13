import argparse, os, json, csv
import torch

from .data import make_loaders
from .models import get_model, iter_prunable_params
from .utils_measure import accuracy, time_inference, model_size_bytes_float_params
from .prune import make_masks_for_model
from .quant import quantize_symmetric_int8, dequantize_int8

def try_import_cuda():
    """Attempt to import the built CUDA extension.

    First try a normal import. If that fails, insert the repo's `cuda_ext`
    directory into sys.path and retry. Return the module or None.
    """
    try:
        import dequant_ext
        return dequant_ext
    except Exception as e1:
        # Try adding the cuda_ext sibling folder to sys.path and retry.
        try:
            import os, sys
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ext_dir = os.path.join(repo_root, "cuda_ext")
            if ext_dir not in sys.path:
                sys.path.insert(0, ext_dir)
            import dequant_ext
            return dequant_ext
        except Exception as e2:
            print("Could not import dequant_ext. CUDA quantization benchmarks will be skipped.")
            print("First import error:", e1)
            print("Retry import error (after adding cuda_ext to sys.path):", e2)
            return None

def clone_from_ckpt(variant, ckpt, device):
    m = get_model(variant).to(device)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    loaded = torch.load(ckpt, map_location="cpu")

    # Unwrap common checkpoint dictionaries
    if isinstance(loaded, dict):
        if 'state_dict' in loaded:
            state_dict = loaded['state_dict']
        elif 'model_state_dict' in loaded:
            state_dict = loaded['model_state_dict']
        else:
            state_dict = loaded
    else:
        state_dict = loaded

    # If stored with DataParallel/module prefix or under 'backbone.' strip common prefixes
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

    # Load non-strictly and print diagnostics if keys are missing/unexpected
    try:
        res = m.load_state_dict(state_dict, strict=False)
        if getattr(res, 'missing_keys', None):
            print(f"Warning: missing keys when loading checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
        if getattr(res, 'unexpected_keys', None):
            print(f"Warning: unexpected keys in checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
    except Exception as e:
        # Fallback: try filtering to model keys only
        if isinstance(state_dict, dict):
            model_keys = set(m.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            res = m.load_state_dict(filtered, strict=False)
            print("Loaded filtered checkpoint (only keys matching model).")
            if getattr(res, 'missing_keys', None):
                print(f"Warning: missing keys when loading filtered checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
            if getattr(res, 'unexpected_keys', None):
                print(f"Warning: unexpected keys in filtered checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
        else:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    m.eval()
    return m

@torch.no_grad()
def apply_prune_inplace(model, sparsity):
    named_weights = list(iter_prunable_params(model))
    masks = make_masks_for_model(named_weights, sparsity=sparsity)
    for name, W in named_weights:
        W.mul_(masks[name].to(dtype=W.dtype))
    # report avg sparsity
    avg_s = sum(float((~m).float().mean().item()) for m in masks.values()) / max(len(masks), 1)
    return masks, avg_s

@torch.no_grad()
def apply_quant_python_inplace(model, masks=None):
    named_weights = list(iter_prunable_params(model))
    scales = {}
    for name, W in named_weights:
        m = masks[name] if masks is not None else torch.ones_like(W, dtype=torch.bool, device=W.device)
        W_eff = W * m.to(dtype=W.dtype)
        Q, s = quantize_symmetric_int8(W_eff)
        W_hat = dequantize_int8(Q, s) * m.to(torch.float16)
        W.data = W_hat.to(dtype=W.dtype)
        scales[name] = float(s)
    return scales

@torch.no_grad()
def apply_quant_cuda_inplace(model, ext, masks=None):
    named_weights = list(iter_prunable_params(model))
    scales = {}
    for name, W in named_weights:
        m = masks[name] if masks is not None else torch.ones_like(W, dtype=torch.bool, device=W.device)
        W_eff = W * m.to(dtype=W.dtype)
        Q, s = quantize_symmetric_int8(W_eff)
        mask_u8 = m.to(torch.uint8)
        W_fp16 = ext.dequant_masked_int8_to_fp16(Q, float(s), mask_u8)
        W.data = W_fp16.to(dtype=W.dtype)
        scales[name] = float(s)
    return scales

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/vgg2_baseline.pt")
    ap.add_argument("--variant", type=str, default="vgg2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch1", type=int, default=1)
    ap.add_argument("--batch2", type=int, default=128)
    ap.add_argument("--skip_cuda", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    _, test_loader = make_loaders(128)
    device = args.device

    x1 = torch.randn(args.batch1, 3, 32, 32, device=device)
    x2 = torch.randn(args.batch2, 3, 32, 32, device=device)

    ext = None if (args.skip_cuda or not device.startswith("cuda")) else try_import_cuda()

    rows = []
    def add_row(name, model, extra=None):
        acc = accuracy(model, test_loader, device=device)
        t1 = time_inference(model, x1, device=device)
        t2 = time_inference(model, x2, device=device)
        size = model_size_bytes_float_params(model)
        d = {"variant": name, "test_acc": float(acc), "t_batch1_s": float(t1), "t_batch128_s": float(t2), "float_param_bytes": int(size)}
        if extra: d.update(extra)
        rows.append(d)

    # Baseline
    base = clone_from_ckpt(args.variant, args.ckpt, device)
    add_row("baseline_fp", base)

    # Prune-only (inplace mask) at a few sparsities
    for s in [0.5, 0.8, 0.9]:
        m = clone_from_ckpt(args.variant, args.ckpt, device)
        masks, avg_s = apply_prune_inplace(m, s)
        add_row(f"prune_only_s{s}", m, {"avg_mask_sparsity": float(avg_s)})

    # Quant-only python
    m = clone_from_ckpt(args.variant, args.ckpt, device)
    scales = apply_quant_python_inplace(m, masks=None)
    add_row("quant_int8_python_all", m, {"num_layers": len(scales)})

    # Prune + quant python
    for s in [0.8, 0.9]:
        m = clone_from_ckpt(args.variant, args.ckpt, device)
        masks, avg_s = apply_prune_inplace(m, s)
        scales = apply_quant_python_inplace(m, masks=masks)
        add_row(f"prune{s}_quant_int8_python_all", m, {"avg_mask_sparsity": float(avg_s), "num_layers": len(scales)})

    # CUDA quant (if available)
    if ext is not None:
        print("Running CUDA quantization benchmarks (dequant in CUDA)...")
        m = clone_from_ckpt(args.variant, args.ckpt, device)
        scales = apply_quant_cuda_inplace(m, ext, masks=None)
        add_row("quant_int8_cuda_dequant_all", m, {"num_layers": len(scales)})

        for s in [0.8, 0.9]:
            m = clone_from_ckpt(args.variant, args.ckpt, device)
            masks, avg_s = apply_prune_inplace(m, s)
            scales = apply_quant_cuda_inplace(m, ext, masks=masks)
            add_row(f"prune{s}_quant_int8_cuda_dequant_all", m, {"avg_mask_sparsity": float(avg_s), "num_layers": len(scales)})

    with open("results/bench.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open("results/bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in sorted(rows[0].keys())})

    prompts = [
        "1) Why doesn’t unstructured mask-based pruning speed up GPU inference in this lab?",
        "2) When pruning ALL layers, which layers are most sensitive to sparsity? Why?",
        "3) What does int8 weight quantization reduce in your pipeline: compute, storage, or memory traffic?",
        "4) Compare Python dequant vs CUDA dequant. What overhead remains and why?",
    ]
    print("\nWrite short answers in answers.md:")
    for p in prompts:
        print(" -", p)

    print("\nWrote results/bench.json and results/bench.csv")

if __name__ == "__main__":
    main()
