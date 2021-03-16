#!/usr/bin/env python
from distutils.core import setup

description = 'Minimalistic implementation of the Self Organizing Maps (SOM)'
keywords = ['machine learning', 'neural networks', 'clustering', 'dimentionality reduction']

setup(name='MiniSom',
      version='2.2.8',
      description=description,
      author='Giuseppe Vettigli',
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="CC BY 3.0",
      py_modules=['minisom'],
      requires=['numpy'],
      url='https://github.com/JustGlowing/minisom',
      download_url='https://github.com/JustGlowing/minisom/archive/master.zip',
      keywords=keywords)
