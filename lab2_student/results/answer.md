1. Unstructured mask pruning does not speed up GPU inference here because the model still runs Conv2d/Linear kernels.
The zeros are multiplied in the same dense math, so FLOPs/kernel shape stay almost the same. Timing confirms this: 
baseline ~0.00513s vs prune s=0.9 ~0.00503s at batch 128, only tiny noise-level change

2. The most sensitive layers are typically the early conv layers and the final layer. Early convs extract fundamental 
features with less redundancy, and the lat linear layer directly maps features to classes, so heavy pruning there
will affects accuracy. In the bench.json, accuracy drops more at higher sparsity and aggressive schedules, sensitivity is strongest in early conv + final classifier, and pruning all layers compounds error. 

3. int8 quantization mainly reduces weight storage. It does not reduce compute in this lab, because weights are 
dequantized back to fp16/fp32 before layer compute, and dense float kernels are still used. 

4. Python dequant vs CUDA dequant: CUDA dequant should reduce dequant overhead by moving it to a parallel kernel and 
avoiding Python loops/tensors ops. But overhead still remains from lauching dequant work per layer, reading int8 + mask 
and writing fp16 weights, and then still doing dense float inference afterward. So dequant can get faster, but end to 
end speedup is limited by remaining memory movement + dense conv/linear compute. 

