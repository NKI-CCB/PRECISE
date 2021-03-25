from setuptools import setup
from setuptools.extension import Extension

import numpy
from numpy.distutils.system_info import get_info


setup(
    name='precise',
    version='1.2',
    description='Patient Response Estimation Corrected by Interpolation of Subspaces Embeddings',
    author='Soufiane Mourragui',
    author_email='s.mourragui@nki.nl',
    url='https://github.com/NKI-CCB/PRECISE',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Computational Biology',
        'Topic :: Scientific/Engineering :: Cancer Research',
        'Topic :: Scientific/Engineering :: Cancer Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.7'
    ],
    ext_modules=[],
    packages=['precise']
)