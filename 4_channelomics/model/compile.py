import numpy as np

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension('ChannelOmniCython_comp',
                        ['ChannelOmniCython_comp.pyx'],
                        include_dirs = [np.get_include()])]

setup(
    ext_modules = cythonize(extensions)
)
