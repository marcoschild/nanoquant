from setuptools import setup, find_packages

setup(
    name='nanoquant',
    version='0.1.0',
    description='Minimal hybrid quantum-classical neural network framework',
    author='Marco',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    python_requires='>=3.7',
)
