[![Documentation Status](https://readthedocs.org/projects/dppy/badge/?version=latest)](https://dppy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/guilgautier/DPPy.svg?token=jftmsjDJSt2JLJqsgR9n&branch=master)](https://travis-ci.com/guilgautier/DPPy)

# DPPy is meant to become a Python library for exact and approximate sampling of Determinantal Point Processes.

> Anything that can go wrong, will go wrong. -- Murphy's Law

## Introduction

Determinantal Point Processes (DPPs) are distributions over sets of items that model diversity using kernels. 
Their applications in machine learning include summary extraction and recommendation systems.
Yet, the cost of sampling from a DPP is prohibitive in large-scale applications, which has triggered an effort towards efficient approximate samplers.

This library aims to provide an implementation of state of the art exact and approximate DPP and k-DPP samplers for both discrete and continuous cases.

## Requirements

DPPy works with [Python 3.4+](https://docs.python.org/3/)

### Dependencies
 - [NumPy](http://www.numpy.org)
 - [SciPy](http://www.scipy.org/)
 - [Matplotlib](http://matplotlib.org/)

The `zono_sampling` mcmc sampler for discrete DPPs requires CVXOPT which itself requires GCC
 - [CVXOPT](http://cvxopt.org)
 - [GCC](http://gcc.gnu.org)
    - On MAC it comes with [Xcode](https://developer.apple.com/xcode/)
    - On UNIX, use your package manager (`apt`, `yum` etc)
        ```bash
        sudo apt install -qq gcc g++
        ```

## Download
### Install from sources

Clone this repository

```bash
git clone https://github.com/guilgautier/DPPy.git
cd DPPy
```

And execute `setup.py`

```bash
pip install .
```

### How to cite this work?
If you use this package for your own work, please consider citing it with this piece of BibTeX:

```bibtex
@misc{DPPy,
    title =   {{DPPy: an Open-Source project for sampling Determinantal Point Processes in Python}},
    author =  {Guillaume Gautier},
    year =    {2018},
    url =     {https://github.com/guilgautier/DPPy/},
    howpublished = {Online at: \url{github.com/guilgautier/DPPy/}},
    note =    {Code at https://github.com/guilgautier/DPPy/, documentation at https://dppy.readthedocs.io/}
}
```