---
title: 'MiniSom, a minimalistic and Numpy based implementation of the Self Organizing Maps'
tags:
  - Python
  - Self Organizing Maps
  - Machine Learning
  - Neural Networks
authors:
  - name: Giuseppe Vettigli
    orcid: 0000-0002-3939-2801
    affiliation: "1, 2, 3"
affiliations:
 - name: Centrica plc (current)
   index: 1
 - name: Cybernetics Institute, Italian National Research Council (previous)
   index: 2
 - name: Parthenope University (previous)
   index: 3
date: 25 April 2021
bibliography: paper.bib
---

# Summary

SOM is a well known type of Artificial Neural Network [@kohonen1990self] that is able to organize itself so that specific areas respond in a similar way to input patterns thats are similar. Since its first formulation, it has been successfully used for a plethora of applications in many scientific fields. The Machine Learning community has found numerous applications and developed a staggering amount of variants of the original model. `MiniSom` is a minimalistic and Numpy [@numpy] based implementation of the Self Organizing Maps (SOM).


# Statement of need

In a scenario where Python has become one of the major languages for scientific development, `MiniSom` serves three main pruposes. First, offer an implementation of SOM in Python which is easy to use and adapt. Second, give researchers the ability to easily create variants of the main SOM model. Third, offer students an implementation of SOM which is easy to understand.

The interface of `MiniSom` has evolved to blend with popular Machine Learning frameworks, as `scikit-learn` [@pedregosa2011scikit], and the visualization library `matplotlib` [@matplotlib]. The documentation of the library is proposed through examples based on `ipython` notebooks [@ipython] and uses the cited libraries.

# Applications 

At the time I am writing, `Minisom` has been cited in more than 50 scientific publications. It has been used in many typical Machine Learning applications, as time series modeling [@fortuin2018som] and text mining [@makiyama2015text]. And it has also been been used as a tool in a variety of fields, as Geophysics [@lessin2020modeling] and Climatology [@thompson2020synoptic]. `MiniSom` has been used for the creation of teaching material for courses at University level and MOOCs, see [@lisbonuni] for an example of teaching material based on `MiniSom`.

# Historical note

`MiniSom` was developed while creating a Machine Learning methodology to embed structured data (graphs and trees) into vectorial spaces [@mythesis; @vettigli2017fuzzy]. The developed has been made while the author was affiliated to institutions 2 and 3.


# References