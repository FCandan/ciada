from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ciada",
    version="1.0.0",
    author="CIADA Research Team",
    description="Crow-Inspired Adaptive Displacement Algorithm — Bitki Besleme Optimizasyonu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ciada-research/ciada",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
