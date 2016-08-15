"""
The dumbest setup.py file in the world.
Last update: 08/15/2016
"""
# these are must haves for cythonizing 
from distutils.core import setup
from Cython.Build import cythonize


setup(
      ext_modules = cythonize("cython_text.pyx")
)
