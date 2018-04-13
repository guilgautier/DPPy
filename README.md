[![Build Status](https://travis-ci.com/guilgautier/DPPy.svg?token=jftmsjDJSt2JLJqsgR9n&branch=master)](https://travis-ci.com/guilgautier/DPPy)

# DPPy is meant to become a Python library for exact and approximate sampling of Determinantal Point Processes.

>Anything that can go wrong, will go wrong. -- Murphy's Law

## Introduction

Determinantal Point Processes (DPPs) are distributions over sets of items that model diversity using kernels. 
Their applications in machine learning include summary extraction and recommendation systems.
Yet, the cost of sampling from a DPP is prohibitive in large-scale applications, which has triggered an effort towards efficient approximate samplers.

This library aims to provide an implementation of state of the art exact and approximate DPP and k-DPP samplers.
For now, only the discrete case is tackled.

## Requirements

DPPy works with Python 2.7+

### Dependencies:
 - [NumPy](http://www.numpy.org)

The `zono_sampling` approximate sampler introduced in [1] requires CVXOPT which itself requires GCC
 - [CVXOPT](http://cvxopt.org)
 - [GCC](http://gcc.gnu.org)
    - On MAC it comes with Xcode
    - On UNIX
        ```bash
        sudo apt-get install -qq gcc g++
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

### Current content

#### Read the Docs

A documentation web page is under development. Since the repo is private, Read the Docs cannot act on the documentation. To see the .html documentation page you can clone the current repository and open the `DPPy/docs/_build/html/index.html` file