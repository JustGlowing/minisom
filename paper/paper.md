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

Self Organizing Maps (SOM) is a type of Artificial Neural Network [@kohonen1990self] that is able to organize itself so that specific areas respond in a similar way to input patterns that are similar. Since its first formulation, it has been successfully used for a plethora of applications in many scientific fields [@kohonen2012self] and the Machine Learning community has developed a staggering amount of variants of the original model. `MiniSom` is a minimalistic and Numpy [@numpy] based implementation of SOM.


# Statement of need

In a scenario where Python has become one of the major languages for scientific development, `MiniSom` serves three main purposes. First, offer an implementation of SOM in Python which is easy to use and adapt. Second, give researchers the ability to easily create variants of the main SOM model. Third, offer students an implementation of SOM which is easy to understand.

The interface of `MiniSom` has evolved to blend with popular Machine Learning frameworks, as `scikit-learn` [@pedregosa2011scikit], and the visualization library `matplotlib` [@matplotlib]. The documentation of the library is proposed through examples based on `ipython` notebooks [@ipython] and uses the cited libraries.

# Applications 

At the time I am writing, `Minisom` has been cited in more than 50 scientific publications[^1]. It has been used in many typical Machine Learning applications, such as time series modeling [@fortuin2018som] and text mining [@makiyama2015text]. And it has also been used as a tool in a variety of fields, such as Geophysics [@lessin2020modeling], Climatology [@thompson2020synoptic], and Network Security [@nam2018self]. Also, `MiniSom` has been used for the creation of teaching material for courses at University level and MOOCs, see [@lisbonuni] for an example of teaching material based on `MiniSom`. I'm also aware of industrial applications of `MiniSom` at TrendMiner[^2].

[^1]: This was estimated via Google Scholar including theses and dissertations.
[^2]: https://www.trendminer.com

# Historical note

`MiniSom` was developed while creating a Machine Learning methodology to embed structured data (graphs and trees) into vectorial spaces [@mythesis; @vettigli2017fuzzy]. The development has been made while the author was affiliated with institutions 2 and 3.


# References