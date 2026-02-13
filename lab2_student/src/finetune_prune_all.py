import argparse, os, json
import torch
import torch.nn as nn
import torch.optim as optim

from .data import make_loaders
from .models import get_model, iter_prunable_params
from .utils_measure import accuracy
from .schedule import sparsity_schedule
from .prune import make_masks_for_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/vgg2_baseline.pt")
    ap.add_argument("--variant", type=str, default="vgg2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lr", type=float, default=1e-5)

    ap.add_argument("--s_target", type=float, default=0.8)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--ramp_end_frac", type=float, default=0.8)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--save_ckpt", action="store_true", help="Save the finetuned/pruned model to checkpoints/")
    ap.add_argument("--out_ckpt", type=str, default=None, help="Optional path to save the pruned checkpoint")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    train_loader, test_loader = make_loaders(args.batch_size)
    device = args.device

    def load_model():
        model = get_model(args.variant).to(device)
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")

        loaded = torch.load(args.ckpt, map_location="cpu")

        # Unwrap common checkpoint dicts
        if isinstance(loaded, dict):
            if 'state_dict' in loaded:
                state_dict = loaded['state_dict']
            elif 'model_state_dict' in loaded:
                state_dict = loaded['model_state_dict']
            else:
                state_dict = loaded
        else:
            state_dict = loaded

        # Strip prefixes from keys (module., backbone.) if present
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

        # Try loading non-strictly and warn about missing/unexpected keys
        try:
            res = model.load_state_dict(state_dict, strict=False)
            if getattr(res, 'missing_keys', None):
                print(f"Warning: missing keys when loading checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
            if getattr(res, 'unexpected_keys', None):
                print(f"Warning: unexpected keys in checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
        except Exception as e:
            # Try filtering to model keys only as a fallback
            if isinstance(state_dict, dict):
                model_keys = set(model.state_dict().keys())
                filtered = {k: v for k, v in state_dict.items() if k in model_keys}
                res = model.load_state_dict(filtered, strict=False)
                print("Loaded filtered checkpoint (only keys matching model).")
                if getattr(res, 'missing_keys', None):
                    print(f"Warning: missing keys when loading filtered checkpoint: {res.missing_keys[:10]}{'...' if len(res.missing_keys)>10 else ''}")
                if getattr(res, 'unexpected_keys', None):
                    print(f"Warning: unexpected keys in filtered checkpoint: {res.unexpected_keys[:10]}{'...' if len(res.unexpected_keys)>10 else ''}")
            else:
                raise RuntimeError(f"Failed to load checkpoint: {e}")

        return model

    def run_one(s_target, warmup_frac, ramp_end_frac):
        model = load_model()
        opt = optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        total_steps = args.epochs * len(train_loader)
        step = 0
        last_masks = None

        for ep in range(args.epochs):
            model.train()
            for x, y in train_loader:
                step += 1
                progress = step / max(total_steps, 1)
                s = sparsity_schedule(progress, s_target=s_target, warmup_frac=warmup_frac, ramp_end_frac=ramp_end_frac)

                # masks for ALL conv/linear weights
                named_weights = list(iter_prunable_params(model))
                masks = make_masks_for_model(named_weights, sparsity=s)
                last_masks = masks

                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                # Apply masks during forward by temporarily masking weights in-place
                # (simple approach for this lab; after forward, restore not needed because we use W*mask via hooks would be nicer)
                # Here we do: save originals -> mask -> forward -> restore originals
                originals = {}
                for n, W in named_weights:
                    originals[n] = W.data.clone()
                    W.data.mul_(masks[n].to(dtype=W.dtype))

                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                # enforce mask after the optimizer step so pruned weights remain zero
                # (restoring originals here would cancel the update we just applied)
                for n, W in named_weights:
                    W.data.mul_(masks[n].to(dtype=W.dtype))

            acc = accuracy(model, test_loader, device=device)
            print(f"finetune epoch {ep+1}/{args.epochs}  s_target={s_target}  test_acc={acc:.4f}")

        # report average sparsity across prunable tensors
        sparsities = []
        for n, W in iter_prunable_params(model):
            m = last_masks[n]
            sparsities.append(float((~m).float().mean().item()))
        out = {
            "s_target": float(s_target),
            "warmup_frac": float(warmup_frac),
            "ramp_end_frac": float(ramp_end_frac),
            "avg_mask_sparsity": float(sum(sparsities)/max(len(sparsities),1)),
            "test_acc": float(accuracy(model, test_loader, device=device)),
        }

        # Optionally save the finetuned/pruned model (state_dict) and masks
        if args.save_ckpt or args.out_ckpt:
            os.makedirs("checkpoints", exist_ok=True)
            if args.out_ckpt:
                outpath = args.out_ckpt
            else:
                outpath = f"checkpoints/{args.variant}.pruned.s{int(s_target*100)}.pth"
            ckpt = {
                'state_dict': model.state_dict(),
                'masks': {k: v.cpu() for k, v in (last_masks.items() if last_masks is not None else [])},
                's_target': float(s_target),
            }
            torch.save(ckpt, outpath)
            print(f"Saved pruned checkpoint: {outpath}")

        return out

    if args.sweep:
        configs = [
            (0.8, 0.0, 0.5),
            (0.8, 0.1, 0.8),
            (0.8, 0.2, 1.0),
            (0.9, 0.0, 0.5),
            (0.9, 0.1, 0.8),
            (0.9, 0.2, 1.0),
        ]
        results = [run_one(*cfg) for cfg in configs]
        with open("results/prune_sweep.json", "w") as f:
            json.dump(results, f, indent=2)
        print("wrote results/prune_sweep.json")
    else:
        out = run_one(args.s_target, args.warmup_frac, args.ramp_end_frac)
        with open("results/prune_one.json", "w") as f:
            json.dump(out, f, indent=2)
        print("wrote results/prune_one.json")

if __name__ == "__main__":
    main()
