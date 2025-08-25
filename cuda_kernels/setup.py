from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='linear_kernel',
    ext_modules=[
        CUDAExtension('linear_kernel', ['linear_kernel.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)