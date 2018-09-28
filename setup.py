# setup.py
from setuptools import setup

setup(
    name='ceres',
    version='0.1.0',
    packages=['ceres'],
    install_requires=[
        'numpy',
        'matplotlib',
        'baselines',
        'gym',
        'h5py',
        'pygame',
        'quadprog',
    ],
)
