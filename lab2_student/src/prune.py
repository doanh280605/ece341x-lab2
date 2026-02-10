import torch

def magnitude_mask(W: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return boolean mask True=keep False=prune with ~sparsity zeros.

    TODO(Pruning-1): implement magnitude thresholding.
      - handle s=0 and s>=1
      - use abs(W).flatten() and kthvalue / quantile logic
      - return mask as torch.bool on same device

    Students should implement this function.
    """
    if sparsity <= 0: 
        return torch.ones_like(W, dtype=torch.bool) # create a boolean mask of all True on the same shape as W

    if sparsity >= 1:
        return torch.zeros_like(W, dtype=torch.bool) # create a boolean mask of all False on the same shape as W

    flat = W.abs().flatten() # flatten the tensor and take absolute values for easy thresholding

    k = int(sparsity * flat.numel()) # calculate the number of elements to prune based on sparsity

    if k == 0: 
        return torch.ones_like(W, dtype=torch.bool) # if no pruning, return all True mask
    if k >= flat.numel(): # if pruning count hits or exceeds total, prune everything
        return torch.zeros_like(W, dtype=torch.bool)

    threshold, _ = torch.kthvalue(flat, k) # find the k-th smallest value which will be the threshold for pruning
    mask = W.abs() > threshold # create a boolean mask where True means keep and False means prune

    return mask


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
