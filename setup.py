#!/usr/bin/env python

from distutils.core import setup

try:
    import numpy
    setup(name='MiniSom',
      version='0.1',
      description='Minimalistic implementation of the Self Organizing Maps (SOM)',
      author='Giuseppe Vettigli',
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="CC BY 3.0",
      py_modules=['minisom'],
      requires = ['numpy']
     )
except ImportError:
	print 'You need Numpy in order to use MiniSom: http://www.scipy.org/install.html'