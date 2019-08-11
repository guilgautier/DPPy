"""A setuptools based setup module.

See:
https://packaging.python.org/tutorials/distributing-packages/#configuring-your-project
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dppy',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.2.0dev1',

    description='DPPy is a Python library for exact and approximate sampling of Determinantal Point Processes.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/guilgautier/DPPy',

    # Author details
    author='Guillaume Gautier',
    author_email='guillaume.gautier@univ-lille1.fr',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        # Specify supported OS.
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],

    # What does your project relate to?
    keywords='Determinantal Point Processes, sampling schemes, random matrices',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.
    # These will be installed by pip when your project is installed. For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy',
                      'scipy>=1.1.0',
                      'matplotlib',
                      'cvxopt==1.2.1',  # For zonotope MCMC sampler
                      'networkx',
                      'sphinxcontrib-bibtex',  # Documentation bibliography
                      'sphinx_rtd_theme'],  # Documentation theme

    project_urls={  # Optional
        "Companion paper": "https://github.com/guilgautier/DPPy_paper",
        "arXiv": "https://arxiv.org/abs/1809.07258",
        "Coverage": "https://coveralls.io/github/guilgautier/DPPy?branch=master",
        "Travis": "https://travis-ci.com/guilgautier/DPPy",
        "Documentation": "https://dppy.readthedocs.io/en/latest/",
        "Source code": "https://github.com/guilgautier/DPPy"
    }
    # List additional groups of dependencies here (e.g. development dependencies). You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
