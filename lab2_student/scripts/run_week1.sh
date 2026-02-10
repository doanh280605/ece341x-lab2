#!/usr/bin/env bash
set -euo pipefail

# Week 1 automation: train (optional), finetune with pruning, export quantized weights (Python path)
# Environment overrides (optional):
#   VARIANT (vgg2), CKPT (input checkpoint), EPOCHS (finetune), BATCH_SIZE, S_TARGET, SWEEP (true/false)

VARIANT=${VARIANT:-vgg2}
CKPT=${CKPT:-checkpoints/vgg-2.cifar.pretrained.pth}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
S_TARGET=${S_TARGET:-0.8}
SWEEP=${SWEEP:-false}

echo "[week1] variant=$VARIANT  ckpt=$CKPT  epochs=$EPOCHS  batch_size=$BATCH_SIZE  s_target=$S_TARGET  sweep=$SWEEP"

# If checkpoint missing, offer to train baseline (this can be long). Only auto-train if TRAIN_IF_MISSING=yes
TRAIN_IF_MISSING=${TRAIN_IF_MISSING:-no}
if [ ! -f "$CKPT" ]; then
	echo "[week1] checkpoint $CKPT not found."
	if [ "$TRAIN_IF_MISSING" = "yes" ]; then
		echo "[week1] training baseline (this may take a while)..."
		python -m src.train_baseline_optional --variant "$VARIANT" --epochs 10 --batch_size "$BATCH_SIZE" --device cuda --out "$CKPT"
	else
		echo "[week1] SKIPPING baseline training. Please provide a checkpoint at $CKPT or set TRAIN_IF_MISSING=yes to auto-train."
	fi
fi

# Run finetune/prune
if [ "$SWEEP" = "true" ] || [ "$SWEEP" = "1" ]; then
	echo "[week1] running finetune sweep..."
	python -m src.finetune_prune_all --sweep --device cuda --variant "$VARIANT" --ckpt "$CKPT"
else
	echo "[week1] running finetune (s_target=$S_TARGET, epochs=$EPOCHS, batch_size=$BATCH_SIZE)..."
	python -m src.finetune_prune_all --device cuda --variant "$VARIANT" --ckpt "$CKPT" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --s_target "$S_TARGET" --save_ckpt
fi

# Determine pruned checkpoint path written by finetune_prune_all when --save_ckpt is used
S_PCT=$(awk -v s="$S_TARGET" 'BEGIN{printf("%d", s*100)}')
PRUNED_CKPT="checkpoints/${VARIANT}.pruned.s${S_PCT}.pth"
if [ -f "$PRUNED_CKPT" ]; then
	echo "[week1] found pruned checkpoint: $PRUNED_CKPT"
	EXPORT_CKPT="$PRUNED_CKPT"
else
	echo "[week1] pruned checkpoint not found at $PRUNED_CKPT. Falling back to input checkpoint: $CKPT"
	EXPORT_CKPT="$CKPT"
fi

# Export quantized weights using Python reference dequant path
echo "[week1] exporting quantized weights (python dequant) using checkpoint: $EXPORT_CKPT"
python -m src.export_quant_all --device cuda --variant "$VARIANT" --ckpt "$EXPORT_CKPT" --sparsity "$S_TARGET"

echo "[week1] done. Results are in results/ (quant export and prune outputs)."
