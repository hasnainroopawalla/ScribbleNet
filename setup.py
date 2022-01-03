from setuptools import setup, find_packages
import os

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, "README.md")) as fh:
    long_description = fh.read()

setup(
    name="scribblenet",
    version="1.0.0",
    packages=find_packages(exclude="tests"),
    description="A Python library that classifies hand-drawn doodles into distinct classes (trained on the QuickDraw dataset by Google).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasnainroopawalla/ScribbleNet",
    author="Hasnain Roopawalla",
    author_email="hasnain.roopawalla@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="python, machinelearning, ai, doodle",
    python_requires=">=3.8",
)
