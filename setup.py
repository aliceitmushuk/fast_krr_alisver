import os
import re
from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_version():
    """Extracts the package version from __init__.py."""
    with open(os.path.join("fast_krr", "__init__.py"), "r", encoding="utf-8") as f:
        match = re.search(r'__version__ = "(.*?)"', f.read())
        return match.group(1) if match else "0.0.0"


setup(
    name="fast_krr",
    version=get_version(),  # Dynamically fetch from __init__.py
    author="Pratik Rathore, Zachary Frangella",
    author_email="pratikr@stanford.edu, zfran@stanford.edu",
    description=(
        "Fast kernel ridge regression using approximate sketch-and-project methods"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pratikrathore8/fast_krr",
    license="MIT",
    keywords=(
        "kernel ridge regression, machine learning, optimization, sketch-and-project"
    ),
    packages=find_packages(
        include=[
            "fast_krr.kernels",
            "fast_krr.models",
            "fast_krr.opts",
            "fast_krr.preconditioners",
        ]
    ),
    package_dir={"fast_krr": "fast_krr"},
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/pratikrathore8/fast_krr",
        "Issue Tracker": "https://github.com/pratikrathore8/fast_krr/issues",
    },
    python_requires=">=3.10",
)
