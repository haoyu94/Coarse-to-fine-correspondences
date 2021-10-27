from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grouping',
    ext_modules=[
        CUDAExtension('grouping_cuda', [
            'src/grouping_api.cpp',
            'src/grouping.cpp',
            'src/grouping_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
