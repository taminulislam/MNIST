"""
Setup script for Fashion MNIST Classification Package
"""

from setuptools import setup, find_packages

with open("README_NEW.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-mnist-classifier",
    version="1.0.0",
    author="Taminul Islam",
    author_email="siu856569517@siu.edu",
    description="A deep learning solution for Fashion MNIST classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taminul/MNIST",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fashion-mnist-train=train:main",
            "fashion-mnist-evaluate=evaluate:main",
            "fashion-mnist-experiments=run_experiments:main",
        ],
    },
)
