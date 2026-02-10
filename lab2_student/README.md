# CIFAR-10: Pruning + Weight-Only Int8 Quantization (All Layers) + CUDA Dequant
**Backbone:** pretrained VGG-style network (checkpoint provided by instructor)

This lab extends pruning + quantization to **all Conv2d and Linear layers** using a **provided pruning schedule** (students only tune rates/targets). Students implement:
- magnitude-based pruning masks (unstructured)
- symmetric int8 weight-only quantization
- one small CUDA kernel: **masked int8 -> fp16 dequant** (works for ANY weight tensor)

We do **not** implement sparse GEMM/conv; pruning is mask-based and compute stays dense.
Quantization is weight-only; we dequantize to fp16 before running the layer.

---

## Repo layout

```
cifar10_prune_quant_cuda_lab_vgg_all_layers/
  requirements.txt
  checkpoints/
    vgg_cifar10.pt          # <-- instructor-provided checkpoint (place here)
  src/
    data.py
    models.py               # VGG wrapper for CIFAR-10
    utils_measure.py
    schedule.py             # Provided schedule (tune params only)
    prune.py                # TODOs: magnitude masks (all layers)
    quant.py                # TODOs: int8 quant/dequant
    train_baseline_optional.py
    finetune_prune_all.py
    export_quant_all.py
    bench_all.py
  cuda_ext/
    build.py
    bindings.cpp
    dequant_cuda.cu         # TODOs: kernel + launcher
    test_correctness.py
  scripts/
    run_week1.sh
    run_week2.sh
```

---

## Setup

```bash
pip install -r requirements.txt
```

The instructor checkpoint at:
```
checkpoints/vgg2_baseline.pt
```

---

## Week 1 (Pruning + Quantization in Python)

### 1) Evaluate pretrained checkpoint
```bash
python -m src.bench_all --device cuda --skip_cuda
```

### 2) Finetune with pruning schedule (ALL layers)
Edit `src/prune.py` TODOs, then run a sweep:
```bash
python -m src.finetune_prune_all --sweep --device cuda
```

### 3) Export / evaluate quantization (ALL layers)
Edit `src/quant.py` TODOs, then:
```bash
python -m src.export_quant_all --device cuda
```

---

## Week 2 (CUDA dequant + benchmark)

### 4) Build CUDA extension
```bash
python cuda_ext/build.py
```

### 5) Correctness test (kernel only)
```bash
python cuda_ext/test_correctness.py
```

### 6) Benchmark all variants (including CUDA dequant)
```bash
python -m src.bench_all --device cuda
```

Outputs:
- `results/bench.json`
- `results/bench.csv`

---

## Deliverables
- `results/bench.json` and `results/bench.csv`
- `answers.md` (short answers; prompts printed by bench script)

---

## What students are expected to change
- `src/prune.py`: implement magnitude mask utilities (all layers)
- `src/quant.py`: implement int8 quant + dequant (per-tensor)
- `cuda_ext/dequant_cuda.cu`: implement kernel + launch

Students will **tune schedule knobs** only:
- `s_target`, `warmup_frac`, `ramp_end_frac`

---

## Notes
- We quantize/prune **Conv2d and Linear weights only** (skip BatchNorm, biases by default).
- Accuracy will drop as sparsity increases; the goal is to understand tradeoffs and overheads.
