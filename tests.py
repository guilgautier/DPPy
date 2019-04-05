#!/usr/bin/env python
# Based on https://github.com/bhargavvader/pycobra/blob/master/tests.py

import matplotlib
import pytest
import sys
import warnings

matplotlib.use('agg')
warnings.filterwarnings("ignore", category=FutureWarning)

pytest.main(['-k-slow', '--cov=dppy'])