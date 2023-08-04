from setuptools import setup
from Cython.Build import cythonize
setup(
    name = 'quasiparticleFunctions',
    ext_modules = cythonize("quasiparticleFunctions.pyx"),
    zip_safe = False,
    )


