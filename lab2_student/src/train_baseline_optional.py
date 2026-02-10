# Optional: if you *don't* have a checkpoint and want to train from scratch.
# Not required for the lab when instructor checkpoint is provided.

import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim

from .data import make_loaders
from .models import get_model
from .utils_measure import accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default="vgg16_bn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="checkpoints/vgg_cifar10.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_loader, test_loader = make_loaders(args.batch_size)
    model = get_model(args.variant).to(args.device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        acc = accuracy(model, test_loader, device=args.device)
        print(f"epoch {ep+1}/{args.epochs} test_acc={acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
