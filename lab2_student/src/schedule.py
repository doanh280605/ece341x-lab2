def sparsity_schedule(progress: float, s_target: float, warmup_frac: float, ramp_end_frac: float) -> float:
    """Provided schedule: tune params only; do NOT modify logic."""
    if progress <= warmup_frac:
        return 0.0
    if progress >= ramp_end_frac:
        return float(s_target)
    t = (progress - warmup_frac) / max(ramp_end_frac - warmup_frac, 1e-8)
    return float(s_target) * float(t)
