from setuptools import setup
from setuptools import find_packages

requirements = [
    'numpy',
    'matplotlib',
    'urllib3',
    'tqdm',
    'pillow',
    'scipy',
]

setup(
    name='perceptron',
    description='Robustness benchmark for deep learning models',
    version='1.0.0.dev',
    url="https://github.com/baidu-advbox/perceptron-benchmark",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
