#!/usr/bin/env python
from distutils.core import setup

description = 'Minimalistic implementation of the Self Organizing Maps (SOM)'
keywords = ['machine learning', 'neural networks',
            'clustering', 'dimentionality reduction']

long_description = 'See the github page https://github.com/JustGlowing/minisom'

setup(name='MiniSom',
      version='2.3.1',
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Giuseppe Vettigli',
      package_data={'': ['Readme.md']},
      include_package_data=True,
      license="MIT",
      py_modules=['minisom'],
      requires=['numpy'],
      url='https://github.com/JustGlowing/minisom',
      download_url='https://github.com/JustGlowing/minisom/archive/master.zip',
      keywords=keywords)
