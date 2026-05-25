import os
import shutil

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extra_compile_args = [
    "-flto",
    "-O3",
    "-ffast-math",
    "-fomit-frame-pointer",
    "-funroll-loops",
]
extra_link_args = [
    "-flto",
    "-O3",
    "-s",
]

if shutil.which("lld") and os.environ.get("CC") == "clang":
    extra_link_args.append("-fuse-ld=lld")

extensions = [
    Extension(
        "spectrum_cython",
        ["spectrum_cython.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="spectroterm",
    packages=[],
    ext_modules=cythonize(extensions, language_level=3),
)
