#!/usr/bin/env python
# This script is based on https://github.com/bhargavvader/pycobra/blob/master/tests.py
# PLEASE pip install pytest-subtests to get a report per test when using self.subTest()


import pytest

import matplotlib
import warnings
matplotlib.use('agg')
warnings.filterwarnings("ignore", category=FutureWarning)

pytest.main(['-k-slow', '--cov=dppy', '--cov-report=term-missing'])
