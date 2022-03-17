#!/usr/bin/env python

from dunamai import Version
from setuptools import find_packages, setup

setup(
    author="Peter Yuen",
    author_email="ppeetteerrsx@gmail.com",
    python_requires=">=3.8",
    description="A simple, typed, commented Pytorch implementation of StyleGAN2.",
    install_requires=[
        x.strip() for x in open("requirements.txt").readlines() if x.strip() != ""
    ],
    license="MIT license",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="stylegan2-torch",
    name="stylegan2-torch",
    packages=find_packages(
        include=["stylegan2_torch", "stylegan2_torch.*"],
        exclude=["docs"],
    ),
    package_data={
        "stylegan2_torch": ["*.cpp", "*.cu"],
    },
    test_suite="tests",
    url="https://github.com/ppeetteerrs/stylegan2-torch",
    version=Version.from_any_vcs().serialize(),
    zip_safe=False,
    options={"bdist_wheel": {"universal": True}},
)
