from setuptools import setup
from setuptools.extension import Extension


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
    install_requires=[
        'joblib==1.0.1',
        'numpy==1.19.5',
        'pandas==1.1.5',
        'scikit-learn==0.24.1',
        'scipy==1.5.4',
        'six==1.15.0'
    ],
    ext_modules=[],
    packages=['precise']
)