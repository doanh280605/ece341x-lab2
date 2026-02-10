#!/usr/bin/env bash
set -euo pipefail

# Week 2 automation: build CUDA extension, test correctness, run full benchmarks
# Optional env vars: VARIANT, CKPT, SKIP_BUILD (set to 1 to skip building), SKIP_TEST (skip correctness test)

VARIANT=${VARIANT:-vgg2}
CKPT=${CKPT:-checkpoints/${VARIANT}.pruned.s80.pth}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_TEST=${SKIP_TEST:-0}

echo "[week2] variant=$VARIANT  ckpt=$CKPT  skip_build=$SKIP_BUILD  skip_test=$SKIP_TEST"

# Try to auto-detect nvcc and set CUDA_HOME (if not set)
if [ -z "${CUDA_HOME:-}" ]; then
	NVCC_PATH=$(which nvcc 2>/dev/null || true)
	if [ -n "$NVCC_PATH" ]; then
		export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
		export PATH="$CUDA_HOME/bin:$PATH"
		export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
		echo "[week2] auto-set CUDA_HOME=$CUDA_HOME"
	else
		echo "[week2] WARNING: nvcc not found on PATH and CUDA_HOME is not set. Building extension may fail." >&2
	fi
fi

if [ "$SKIP_BUILD" != "1" ]; then
	echo "[week2] building CUDA extension..."
	python cuda_ext/build.py
else
	echo "[week2] skipping build as requested"
fi

if [ "$SKIP_TEST" != "1" ]; then
	echo "[week2] running CUDA correctness test..."
	python cuda_ext/test_correctness.py
else
	echo "[week2] skipping correctness test as requested"
fi

echo "[week2] running benchmark (this may take several minutes)..."
python -m src.bench_all --device cuda --variant "$VARIANT" --ckpt "$CKPT"

echo "[week2] done. Bench results in results/bench.json and results/bench.csv"
