import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extensions = [CUDAExtension(name="atss_cuda.ops",
                            sources=glob.glob("./atss_cuda/src/*"),
                            include_dirs=[os.path.abspath("./atss_cuda/src")]),]

setup(
    name='atss_cuda',
    version="1.0.0",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
)
