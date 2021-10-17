from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(
        Extension(
            "compute_overlap",
            sources=["compute_overlap.pyx"], 
            include_dirs=[numpy.get_include()]
        )
    ),
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)