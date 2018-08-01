[![Documentation Status](https://readthedocs.org/projects/dppy/badge/?version=latest)](https://dppy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/guilgautier/DPPy.svg?branch=master)](https://travis-ci.com/guilgautier/DPPy)

# DPPy is a Python library for sampling Determinantal Point Processes.

> Anything that can go wrong, will go wrong. − [Murphy's Law](http://phdcomics.com/comics/archive.php?comicid=1867)

## Introduction

Determinantal point processes (DPPs) are specific probability distributions over clouds of points that have been popular as models or computational tools across physics, probability, statistics, and more recently of booming interest in machine learning. Sampling from DPPs is a nontrivial matter, and many approaches have been proposed. DPPy is a Python library that puts together all exact and approximate sampling algorithms for DPPs.

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