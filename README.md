[![Documentation Status](https://readthedocs.org/projects/dppy/badge/?version=latest)](https://dppy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/guilgautier/DPPy.svg?branch=master)](https://travis-ci.com/guilgautier/DPPy)

# DPPy is a Python library for sampling Determinantal Point Processes.

> Anything that can go wrong, will go wrong. − [Murphy's Law](http://phdcomics.com/comics/archive.php?comicid=1867)

## Introduction

Determinantal point processes (DPPs) are specific probability distributions over clouds of points that have been popular as models or computational tools across physics, probability, statistics, and more recently of booming interest in machine learning. 
Sampling from DPPs is a nontrivial matter, and many approaches have been proposed. 
DPPy is a Python library that puts together all exact and approximate sampling algorithms for DPPs.

## Requirements

DPPy works with [Python 3.4+](http://docs.python.org/3/)

### Dependencies
 - [NumPy](http://www.numpy.org)
 - [SciPy](http://www.scipy.org/)
 - [Matplotlib](http://matplotlib.org/)
 - [Networkx](http://networkx.github.io/)

The `zono_sampling` mcmc sampler for finite DPPs requires CVXOPT which itself requires GCC
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

We plan to submit [DPPy](https://github.com/guilgautier/DPPy) to the Machine Learning Open Source Software [MLOSS](http://www.jmlr.org/mloss/) track of JMLR.

If you use this package, please consider citing it with this piece of BibTeX:

```bibtex
@misc{DPPy,
    title =   {{DPPy: Sampling Determinantal Point Processes with Python}},
    author =  {Guillaume Gautier, Rémi Bardenet, Michal Valko},
    year =    {2018},
    url =     {https://github.com/guilgautier/DPPy/},
    howpublished = {Online at: \url{github.com/guilgautier/DPPy/}},
    note =    {Code at https://github.com/guilgautier/DPPy/, documentation at https://dppy.readthedocs.io/}
}
```

## Reproducibility

[DPPy](https://github.com/guilgautier/DPPy) would have never existed without [Guillermo Polito](https://github.com/guillep) who is leading the [reproducible work group](https://github.com/CRIStAL-PADR/reproducible-research-SE-notes).
Take a look at the corresponding [booklet](https://github.com/CRIStAL-PADR/reproducible-research-SE-notes) to learn more on how to make your research reproducible!