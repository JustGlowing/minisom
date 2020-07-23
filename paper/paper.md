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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Centrica
   index: 1
 - name: Institute of Applied Sciences and Intelligent Systems “Eduardo Caianiello” of the Italian National Research Council
   index: 2
date: 23 July 2020
bibliography: paper.bib
---

# Summary

`MiniSom` is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.


# Statement of need 

SOM are a well known type of Neural Network [kohonen1990self]. Since its first formulation it has been used as tool in a pletora of scientific fields and the Machine Learning community has not only found noumerous applications for it but has developed a staggering amount of variants of the original model.

`MiniSom` is Python library that implements SOM and it is designed to be easy to modify and adapt. The goal is to give reseacher the ability to easily create variants of the main SOM model and give to students an easy to understand implementation of SOM.

`Minisom` has already been currently cited by more than 50 documents including academic papers and theses. It has been used in many typcal Machine Learning applications, as time series modelling [@fortuin2018som] and text mining [@makiyama2015text]. And it has also been using in other fields as Geophysics [@lessin2020modeling] and Climatology [@thompson2020synoptic]. The paper was originally developed while developing on a methodology to embed structured data into vectorial spaces [@vettigli2017fuzzy].

The documentation of `MiniSom` shows how it can be used in combination with more general MAchine Learning framework `scikit-learn` [@pedregosa2011scikit]. Also, `MiniSom` has been used for the creation of teaching materials on Machine Learning. See [@author:2001] for an example of didactic material made using `MiniSom`.


# References