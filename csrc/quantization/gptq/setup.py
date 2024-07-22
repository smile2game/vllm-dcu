from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17","-DUSE_ROCM","-U__HIP_NO_HALF_CONVERSIONS__","-U__HIP_NO_HALF_OPERATORS__"]
#--gpu-max-threads-per-block=1024编译会导致GPTQ多batch性能下降。
# NVCC_FLAGS = ["-O3", "-std=c++17","-DUSE_ROCM","--gpu-max-threads-per-block=1024","-U__HIP_NO_HALF_CONVERSIONS__","-U__HIP_NO_HALF_OPERATORS__"]


ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

extra_compile_args={
    "cxx": CXX_FLAGS,
    "nvcc": NVCC_FLAGS,
}

setup(
    name="gptq_kernels",
    ext_modules=[
        CUDAExtension(
            name="gptq_kernels",
            sources=[
                "./torch_bindings.cpp",
                "./q_gemm.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
