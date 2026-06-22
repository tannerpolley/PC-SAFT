import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


PACKAGE_ROOT = "src/pcsaft"

extra_compile_args = []
if os.name == "nt":
    extra_compile_args.extend(["/std:c++17", "/wd4551"])
else:
    extra_compile_args.append("-std=c++17")

ext_modules = [
    Extension(
        "pcsaft.pcsaft",
        sources=[f"{PACKAGE_ROOT}/pcsaft.pyx"],
        language="c++",
        include_dirs=[
            np.get_include(),
            PACKAGE_ROOT,
            "externals/eigen",
            "externals/autodiff",
        ],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
    )
)

