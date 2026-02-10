import torch

def magnitude_mask(W: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return boolean mask True=keep False=prune with ~sparsity zeros.

    TODO(Pruning-1): implement magnitude thresholding.
      - handle s=0 and s>=1
      - use abs(W).flatten() and kthvalue / quantile logic
      - return mask as torch.bool on same device

    Students should implement this function.
    """
    raise NotImplementedError("Implement magnitude_mask in student version")


def make_masks_for_model(named_weights, sparsity: float):
    """Create a dict: name -> bool mask for each weight tensor.
    named_weights: iterable of (name, weight_tensor)
    """
    masks = {}
    for name, W in named_weights:
        masks[name] = magnitude_mask(W, sparsity)
    return masks


def apply_mask_(W: torch.Tensor, mask: torch.Tensor):
    """In-place masking (used only in eval/export paths)."""
    W.mul_(mask.to(dtype=W.dtype))
