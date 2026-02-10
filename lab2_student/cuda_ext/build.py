import os
from torch.utils.cpp_extension import load

this_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(this_dir, "bindings.cpp"),
           os.path.join(this_dir, "dequant_cuda.cu")]

dequant_ext = load(
    name="dequant_ext",
    sources=sources,
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    build_directory=this_dir,
    verbose=True,
)
print("Built dequant_ext successfully.")
