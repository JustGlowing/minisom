<h1>MiniSom<img src='https://3.bp.blogspot.com/-_6UDGEHzIrs/WSfiyjmoeRI/AAAAAAAABHw/3UQylcCBEhUfHNhf56WSHBBmQ6g_lXQhwCLcB/s320/minisom_logo.png' align='right'></h1>

Self Organizing Maps
--------------------

MiniSom is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

Installation
---------------------

Just use pip:

    pip install minisom

or download MiniSom to a directory of your choice and use the setup script:

    python setup.py install

How to use it
---------------------

In order to use MiniSom you need your data organized as a Numpy matrix where each row corresponds to an observation or as list of lists like the following:

```python
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]      
```

 Then you can run MiniSom just as follows:

```python
from minisom import MiniSom    
som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train_random(data, 100) # trains the SOM with 100 iterations
```

MiniSom implements two types of training. The random training (implemented by the method `train_random`), where the model is trained picking random samples from your data, and the batch training (implemented by the method `train_batch`), where the samples are picked in the order they are stored.

The weights of the network are randmly initialized by default. Two additional methods are provided to initialize the weights in a data driven fashion: `random_weights_init` and `pca_weights_init`.

### Using the trained SOM

After the training you will be able to

* Compute the coordinate assigned to an observation `x` on the map with the method `winner(x)`.
* Compute the average distance map of the weights on the map with the method `distance_map()`.
* Compute the number of times that each neuron have been considered winner for the observations of a new data set with the method `activation_response(data)`.
* Compute the quantization error with the method `quantization_error(data)`.

#### Vector quantization

The data can be quantized by assigning a code book (weights vector of the winning neuron) to each sample in data. This kind of vector quantization is implemented by the method `quantization` that can be called as follows:

```python
qnt = som.quantization(data)
```

In this example we have that `qnt[i]` is the quantized version of `data[i]`.

#### Export a SOM and load it again

A model can be saved using pickle as follows

```python
import pickle
som = MiniSom(7, 7, 4)

# ...train the som here

# saving the some in the file som.p
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
```

and can be loaded as follows

```python
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
```

Note that if a lambda function is used to define the decay factor MiniSom will not be pickable anymore.

Examples
---------------------
The code that produces the following figures is in this notebook: https://github.com/JustGlowing/minisom/blob/master/examples/examples.ipynb

1. Iris flower <a href="http://en.wikipedia.org/wiki/Iris_flower_data_set">dataset</a>.

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_iris.png" alt="Iris example">

For each observation we have a marker placed on the position of the winning neuron on the map. Each type of marker represents a class of the iris data. The average distance map of the weights is used as background.

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_iris_pies.png">

Each neuron is associated with one of the labels in the dataset with a specific degree.

2. Color quantization

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_color_quantization.png" alt="Color quantization example">

3. Images clustering

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_digts.png" alt="handwritten digitts recognition">

The graph above represent each image with the handwritten digit it contains. The position corresponds to the position of the winning neuron for the image. Here we also have a version of this graphs that shows the original images:

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_digts_imgs.png" alt="handwritten digitts recognition" width=500>

4. Outliers detection

<img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_outliers_detection.png" width=550>


Video tutorials
------------

The following video tutorials made by the GeoEngineerings School show how to use MiniSom to build a fraud detection system:

- <a href="https://www.youtube.com/watch?v=3osKNPyAxPM&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=11">Part 1</a>
- <a href="https://www.youtube.com/watch?v=uUpQ6MITlVs&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=12">Part 2</a>
- <a href="https://www.youtube.com/watch?v=mryFU0TEInk&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=13">Part 3</a>
- <a href="https://www.youtube.com/watch?v=9MzFOIoxxdk&index=14&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC">Part 4</a>

Other tutorials
------------
- <a href="https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d">Introduction to Self-Organizing by Derrick Mwiti</a>
- <a href="https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html">Self Organizing Maps on the Glowing Python</a>

Who uses Minisom?
------------
<ul>
<li>
Chintan Shah, Anjali Jivani. <a href="https://ieeexplore.ieee.org/abstract/document/8554848">A Hybrid Approach of Text Summarization Using Latent Semantic Analysis and Deep Learning</a>. 2018 International Conference on Advances in Computing, Communications and Informatics (ICACCI), 2018.
</li>
<li>
Katsutoshi Masai, Kai Kunze, Yuta Sugiura, Maki Sugimoto. <a href="https://dl.acm.org/citation.cfm?id=3267562">Mapping Natural Facial Expressions Using Unsupervised Learning and Optical Sensors on Smart Eyewear</a>. Proceedings of the 2018 ACM International Joint Conference and 2018 International Symposium on Pervasive and Ubiquitous Computing and Wearable Computers, 2018 ACM.
</li>
<li>
Ili Ko, Desmond Chambers, Enda Barrett. <a href="https://ieeexplore.ieee.org/document/8343119/">A Lightweight DDoS Attack Mitigation System within the ISP Domain Utilising Self-organizing Map</a>. Proceedings of the Future Technologies, 2018 Springer.
</li>
<li>
T. M. Nam et al. <a href="https://ieeexplore.ieee.org/document/8343119/">Self-organizing map-based approaches in DDoS flooding detection using SDN</a>. 2018 International Conference on Information Networking (ICOIN), 2018.
</li>
<li>
Li Yuan <a href="http://digitalcommons.uri.edu/theses/1244/">Implementation of Self-Organizing Maps with Python</a>. Master Thesis, University of Rhode Island, 2018.
</li>
<li>
Ying Xie, Linh Le, Yiyun Zhou, Vijay V.Raghavan. <a href="https://www.sciencedirect.com/science/article/pii/S0169716118300026">Deep Learning for Natural Language Processing</a>. Elsevier Handbook of Statistics, 2018.
</li>
<li>
Vincent Fortuin, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. <a href="https://arxiv.org/pdf/1806.02199.pdf">Deep Self-Organization: Interpretable Discrete
Representation Learning on Time Series</a>. 2018.
</li>
<li>
Birgitta Dresp-Langley, John Mwangi Wandeto, Henry Okola Nyongesa. <a href="https://www.openscience.fr/IMG/pdf/iste_fromd2d17v1n2.pdf">Using the quantization error from Self‐Organizing Map (SOM) output for fast detection of critical variations in image time series</a>. ISTE OpenScience, 2018.
</li>
<li>
John M. Wandeto, Henry O. Nyongesa, Birgitta Dresp-Langley. <a href="https://arxiv.org/abs/1803.11125">Detection of Structural Change in Geographic Regions of Interest by Self Organized Mapping: Las Vegas City and Lake Mead across the Years</a>. 2018.
</li>
<li>
Denis Mayr Lima Martins, Gottfried Vossen, Fernando Buarque de Lima Neto. <a href="https://ieeexplore.ieee.org/abstract/document/8285698/">Learning database queries via intelligent semiotic machines</a>. IEEE Latin American Conference on Computational Intelligence (LA-CCI), 2017.
</li>
<li>
Udemy online course. <a href="https://www.udemy.com/deeplearning/?utm_campaign=email&utm_source=sendgrid.com&utm_medium=email">Deep Learning A-Z™: Hands-On Artificial Neural Networks</a>
<img src="https://www.udemy.com/staticx/udemy/images/v6/logo-coral.svg" height="20">
</li>
<li>
Fredrik Broch Elgaaen, Nicholas Mowatt Larssen. <a href="https://www.duo.uio.no/bitstream/handle/10852/57507/master_nml.pdf?sequence=1&isAllowed=y">Data mining i banksektoren - Prediksjonsmodellering og analyse av kunder som sier opp boliglån</a>. University of Oslo, May 2017.
</li>
<li>
Óscar Clavería González, Enric Monte Moreno, Salvador Torra Porras. <a href="http://www.sciencedirect.com/science/article/pii/S2110701715000694">A self-organizing map analysis of survey-based agents׳ expectations before impending shocks for model selection: The case of the 2008 financial crisis</a>. International Economics Volume 146, Pages 40–58. August 2016.
</li>
<li>
Sameen Mansha, Faisal Kamiran, Asim Karim, Aizaz Anwar. <a href="http://link.springer.com/chapter/10.1007/978-3-319-46681-1_16">A Self-Organizing Map for Identifying InfluentialCommunities in Speech-based Networks</a>. Proceeding CIKM '16 Proceedings of the 25th ACM International on Conference on Information and Knowledge Management Pages 1965-1968. 2016.
</li>
<li>
Sameen Mansha, Zaheer Babar, Faisal Kamiran, Asim Karim. <a href="http://link.springer.com/chapter/10.1007/978-3-319-46681-1_16">Neural Network Based Association Rule Mining from Uncertain Data</a>. Neural Information Processing Volume 9950 of the series Lecture Notes in Computer Science pp 129-136. 2016.
</li>
<li>
Makiyama, Vitor Hirota, M. Jordan Raddick, and Rafael DC Santos. <a href="http://ceur-ws.org/Vol-1478/paper7.pdf">Text Mining Applied to SQL Queries: A Case Study for the SDSS SkyServer</a>. 2nd Annual International Symposium on Information Management and Big Data. 2015.
</li>
<li>
Remi Domingues. <a href="http://www.diva-portal.org/smash/get/diva2:897808/FULLTEXT01.pdf">Machine Learning for
Unsupervised Fraud Detection</a>. Royal Institute of Technology School of Computer Science and Communication KTH CSC. 2015.
</li>
<li>Ivana Kajić, Guido Schillaci, Saša Bodiroža, Verena V. Hafner, <a href="http://dl.acm.org/citation.cfm?id=2559816">Learning hand-eye coordination for a humanoid robot using SOMs</a>. Proceedings of the 2014 ACM/IEEE international conference on Human-robot interaction
Pages 192-193.</li>
</ul>

Compatibility notes
---------------------
Minisom has been tested under Python 3.6.2.

License
---------------------

MiniSom by Giuseppe Vettigli is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
