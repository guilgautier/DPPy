DPPy: Sampling Determinantal Point Processes with Python
========================================================

|Documentation Status| |Build Status| |Coverage Status|

.. |Documentation Status| image:: https://readthedocs.org/projects/dppy/badge/?version=latest
   :target: https://dppy.readthedocs.io/en/latest/?badge=latest

.. |Build Status| image:: https://travis-ci.com/guilgautier/DPPy.svg?branch=master
   :target: https://travis-ci.com/guilgautier/DPPy

.. |Coverage Status| image:: https://coveralls.io/repos/github/guilgautier/DPPy/badge.svg?branch=master
  :target: https://coveralls.io/github/guilgautier/DPPy?branch=master

.. |Google Colab| image:: https://badgen.net/badge/Launch/on%20Google%20Colab/blue?icon=terminal
   :target: https://colab.research.google.com/github/guilgautier/DPPy/blob/master/notebooks/Tuto_DPPy.ipynb

*"Anything that can go wrong, will go wrong"* − `Murphy's Law <http://phdcomics.com/comics/archive.php?comicid=1867>`_

Introduction
------------

Determinantal point processes (DPPs) are specific probability
distributions over clouds of points that have been popular as models or
computational tools across physics, probability, statistics, and more
recently of booming interest in machine learning. Sampling from DPPs is
a nontrivial matter, and many approaches have been proposed. DPPy is a
`Python <https://www.python.org/>`__ library that puts together all
exact and approximate sampling algorithms for DPPs.

Requirements
------------

DPPy works with `Python 3.4+ <http://docs.python.org/3/>`__.

Dependencies
~~~~~~~~~~~~

-  `NumPy <http://www.numpy.org>`__
-  `SciPy <http://www.scipy.org/>`__
-  `Matplotlib <http://matplotlib.org/>`__
-  `Networkx <http://networkx.github.io/>`__ to play with `uniform
   spanning
   trees <https://dppy.readthedocs.io/en/latest/exotic_dpps/index.html#uniform-spanning-trees>`__
-  `CVXOPT <http://cvxopt.org>`__ to use the ``zono_sampling`` MCMC
   sampler for finite DPPs. **CVXOPT itself requires**
   `GCC <http://gcc.gnu.org>`__,

   -  On MAC it comes with
      `Xcode <https://developer.apple.com/xcode/>`__
   -  On UNIX, use your package manager (``apt``, ``yum`` etc)

      .. code:: bash

          sudo apt install -qq gcc g++

Installation
------------

DPPy is now available on `PyPI <https://pypi.org/project/dppy/>`__

.. code:: bash

  pip install dppy

However you may not work with the latest version, so

1. If you have a GitHub account

   -  Please consider forking DPPy
   -  Use git to clone your copy of the repo

      .. code:: bash

          cd <directory_of_your_choice>
          git clone https://github.com/<username>/DPPy.git

2. If you only use git, clone this repository

   .. code:: bash

       cd <directory_of_your_choice>
       git clone https://github.com/guilgautier/DPPy.git

3. Otherwise simply dowload the project

4. In any case, install the project with

   .. code:: bash

       cd DPPy
       pip install .

Tutorials in `Jupyter notebooks <https://github.com/guilgautier/DPPy/blob/master/notebooks/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can read and work on these interactive tutorial `Notebooks <https://github.com/guilgautier/DPPy/blob/master/notebooks/>`_, directly from your
web browser, without having to download or install Python or anything.
Just click, wait a little bit, and play with the notebook!

Contribute to the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`documentation <http://dppy.readthedocs.io/>`__
is generated locally with
`Sphinx <http://www.sphinx-doc.org/en/master/>`__ and then built online
by `ReadTheDocs <https://readthedocs.org/projects/dppy/>`__.

If you wish to contribute to the documentation or just play with it
locally, you can:

-  Install Sphinx

   .. code:: bash

       pip install -U sphinx

-  Generate the docs locally

   .. code:: bash

       cd DPPy/docs
       make html

-  Open the local HTML version of the documentation located at
   ``DPPy/docs/_build/html/index.html``

   .. code:: bash

       open _build/html/index.html

How to cite this work?
~~~~~~~~~~~~~~~~~~~~~~

We wrote a companion paper to
`DPPy <https://github.com/guilgautier/DPPy>`__ for latter submission to
the `MLOSS <http://www.jmlr.org/mloss/>`__ track of JMLR.

The companion paper is available on

-  `arXiv <http://arxiv.org/abs/1809.07258>`__ (maybe not upto date)
-  `GitHub <https://github.com/guilgautier/DPPy_paper>`__ for the lastest version

If you use this package, please consider citing it with this piece of
BibTeX:

.. code:: bibtex

  @article{GaBaVa18,
    archivePrefix = {arXiv},
    arxivId = {1809.07258},
    author = {Gautier, Guillaume and Bardenet, R{\'{e}}mi and Valko, Michal},
    eprint = {1809.07258},
    journal = {ArXiv e-prints},
    title = {{DPPy: Sampling Determinantal Point Processes with Python}},
    keywords = {Computer Science - Machine Learning, Computer Science - Mathematical Software, Statistics - Machine Learning},
    url = {http://arxiv.org/abs/1809.07258},
    year = {2018},
    note = {Code at http://github.com/guilgautier/DPPy/ Documentation at http://dppy.readthedocs.io/}
  }

Reproducibility
---------------

We would like to thank `Guillermo Polito <https://guillep.github.io/>`__
for leading our reproducible research
`workgroup <https://github.com/CRIStAL-PADR/reproducible-research-SE-notes>`__,
this project owes him a lot.

Take a look at the corresponding
`booklet <https://github.com/CRIStAL-PADR/reproducible-research-SE-notes>`__
to learn more on how to make your research reproducible!
