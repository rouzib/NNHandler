"""
Setup script for NNHandler package.
"""
from setuptools import setup, find_packages
import os

# Try to read README.rst first (for PyPI), fall back to README.md
if os.path.exists("README.rst"):
    with open("README.rst", "r", encoding="utf-8") as fh:
        long_description = fh.read()
        long_description_content_type = "text/x-rst"
else:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
        long_description_content_type = "text/markdown"

setup(
    name="NNHandler",
    version="0.2.2",
    author="Nicolas Payot",
    author_email="",
    description="A comprehensive framework for PyTorch neural network handling",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url="https://github.com/rouzib/NNHandler",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Pytorch",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "full": [
            "torch_ema",
            "matplotlib",
            "tqdm",
        ],
    },
)
