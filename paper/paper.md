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
    affiliation: 1
affiliations:
 - name: Centrica plc
   index: 1
date: 23 July 2020
bibliography: paper.bib
---

# Summary

`MiniSom` is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.


# Statement of need 

SOM is a well known type of Artificial Neural Network [@kohonen1990self] that is able to organize itself so that each specific areas respond to similarly to similar outputs. This model is suitable for different Machine Learning tasks, specifically clustering, dimensionality reduction and vector quantization. Since the first formulation, it became a tool in a plethora of applications in many scientific fields.  The Machine Learning community has not only found numerous applications for it but has developed a staggering amount of variants of the original model.

`MiniSom` is a Python library that implements SOM and it is designed to be easy to modify and adapt. The goal is to give researchers the ability to easily create variants of the main SOM model and give students an implementation of SOM which is easy to understand.

At the time I am writing, `Minisom` has been cited in more than 50 scientific publications. It has been used in many typical Machine Learning applications, as time series modeling [@fortuin2018som] and text mining [@makiyama2015text]. And it has also been been used as a tool in a variety of fields as Geophysics [@lessin2020modeling] and Climatology [@thompson2020synoptic]. It's also worth mentioning that `MiniSom` has been used for the creation of teaching materials on Machine Learning, see [@lisbonuni] for an example.

The interface of `MiniSom` has evolved to blend easily with the general Machine Learning framework `scikit-learn` [@pedregosa2011scikit] and the visualization library `matplotlib` [@matplotlib]. The documentation of the library is proposed through examples and makes heavily use of the cited libraries.

The library was originally developed while developing a Machine Learning methodology to embed structured data (graphs and trees) into vectorial spaces [@vettigli2017fuzzy].


# References