#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    author="Slaven Glumac",
    author_email='slaven.glumac@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    description=' '.join([
        "A package which demostrates the use of SDF as the model",
        "of computation for a non-iterative co-simulation."
    ]),
    install_requires=[
        'fmpy>=0.2.12',
        'sympy>=1.3',
        'numpy>=1.15.4',
        'matplotlib>=3.0.2',
    ],
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sdf4sim',
    name='sdf4sim',
    packages=find_packages(),
    setup_requires=[
        'pytest-runner',
    ],
    test_suite='tests',
    tests_require=[
        'pytest',
    ],
    url='https://github.com/sglumac/sdf4sim',
    version='0.4.0',
    zip_safe=False,
)
