DPPy: Sampling Determinantal Point Processes with Python
========================================================

|Documentation Status| |Build Status| |Coverage Status| |PyPI package|

.. |Documentation Status| image:: https://readthedocs.org/projects/dppy/badge/?version=latest
  :target: https://dppy.readthedocs.io/en/latest/?badge=latest

.. |Build Status| image:: https://travis-ci.com/guilgautier/DPPy.svg?branch=master
  :target: https://travis-ci.com/guilgautier/DPPy

.. |Coverage Status| image:: https://coveralls.io/repos/github/guilgautier/DPPy/badge.svg?branch=master
  :target: https://coveralls.io/github/guilgautier/DPPy?branch=master

.. |PyPI package| image:: https://img.shields.io/pypi/v/dppy?color=blue
  :target: https://pypi.org/project/dppy/

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

Installation
------------

DPPy works with `Python 3.6+ <http://docs.python.org/3/>`__.

Dependencies
~~~~~~~~~~~~

This project depends on the following libraries, which are automatically downloaded during installation:

-  `NumPy <http://www.numpy.org>`__
-  `SciPy <http://www.scipy.org/>`__
-  `Matplotlib <http://matplotlib.org/>`__

The following dependencies are optional, and unlock extra functionality if installed:

-  `Networkx <http://networkx.github.io/>`__ to play with `uniform
   spanning
   trees <https://dppy.readthedocs.io/en/latest/exotic_dpps/index.html#uniform-spanning-trees>`__ .
-  `CVXOPT <http://cvxopt.org>`__ to use the ``zono_sampling`` MCMC
   sampler for finite DPPs. **CVXOPT itself requires**
   `GCC <http://gcc.gnu.org>`__,

   -  On MAC it comes with
      `Xcode <https://developer.apple.com/xcode/>`__
   -  On UNIX, use your package manager (``apt``, ``yum`` etc)

      .. code:: bash

          sudo apt install -qq gcc g++

-  `Sphinx <http://www.sphinx-doc.org/en/master/>`__ to modify and rebuild the documentation

Installation instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

DPPy is now available on `PyPI <https://pypi.org/project/dppy/>`__ |PyPI package|

.. code:: bash

  pip install dppy

For a full installation, including extra dependencies and dependencies necessary to build the documentation (see above), use :code:`pip install dppy['zonotope','trees','docs']`.

Note that only stable DPPy releases are available on PyPI, and recently included improvements might only be available on the git master branch. If you want to work with the latest version of DPPy

1. If you have a GitHub account please consider forking DPPy and use git to clone your copy of the repo

   .. code:: bash

       cd <directory_of_your_choice>
       git clone https://github.com/<username>/DPPy.git

2. If you only use git, clone this repository

   .. code:: bash

       cd <directory_of_your_choice>
       git clone https://github.com/guilgautier/DPPy.git

3. Otherwise simply download the project

4. In any case, install the project with

   .. code:: bash

       cd DPPy
       pip install .

Use :code:`pip install .['zonotope','trees','docs']` to perform a full install from a local source tree.

How to use it
-------------

The main DPPy documentation is available online at `http://dppy.readthedocs.io <http://dppy.readthedocs.io>`_.
There are also some interactive tutorials using Jupyter available at https://github.com/guilgautier/DPPy/blob/master/notebooks/.
For more details, check below.

Tutorials in `Jupyter notebooks <https://github.com/guilgautier/DPPy/blob/master/notebooks/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can read and work on these interactive tutorial `Notebooks <https://github.com/guilgautier/DPPy/blob/master/notebooks/>`_, directly from your
web browser, without having to download or install Python or anything.
Just click, wait a little bit, and play with the notebook!

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`documentation <http://dppy.readthedocs.io/>`__
is generated locally with
`Sphinx <http://www.sphinx-doc.org/en/master/>`__ and then built online
by `ReadTheDocs <https://readthedocs.org/projects/dppy/>`__.
If you wish to contribute to the documentation or just play with it
locally, you can install the necessary dependencies and then:

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

`We wrote a companion paper to DPPy which got accepted for publication in the Machine Learning Open Source Software track of JMLR <http://jmlr.org/papers/v20/19-179.html>`__.

If you use the DPPy toolbox, please consider citing it with this piece of
BibTeX:

.. code:: bibtex

  @article{GPBV19,
    author = {Gautier, Guillaume and Polito, Guillermo and Bardenet, R{\'{e}}mi and Valko, Michal},
    journal = {Journal of Machine Learning Research - Machine Learning Open Source Software (JMLR-MLOSS)},
    title = {{DPPy: DPP Sampling with Python}},
    keywords = {Computer Science - Machine Learning, Computer Science - Mathematical Software, Statistics - Machine Learning},
    url = {http://jmlr.org/papers/v20/19-179.html},
    year = {2019},
    archivePrefix = {arXiv},
    arxivId = {1809.07258},
    note = {Code at http://github.com/guilgautier/DPPy/ Documentation at http://dppy.readthedocs.io/}
  }

Many of the algorithms implemented in DPPy also have associated literature that you should consider citing.
Please refer to the `bibliography on the documentation page <https://dppy.readthedocs.io/en/latest/bibliography/>`__ and docstrings of each specific DPP sampler for details.

Reproducibility
---------------

We would like to thank `Guillermo Polito <https://guillep.github.io/>`__
for leading our reproducible research
`workgroup <https://github.com/CRIStAL-PADR/reproducible-research-SE-notes>`__,
this project owes him a lot.

Take a look at the corresponding
`booklet <https://github.com/CRIStAL-PADR/reproducible-research-SE-notes>`__
to learn more on how to make your research reproducible!
