<h1>MiniSom<img src='https://3.bp.blogspot.com/-_6UDGEHzIrs/WSfiyjmoeRI/AAAAAAAABHw/3UQylcCBEhUfHNhf56WSHBBmQ6g_lXQhwCLcB/s320/minisom_logo.png' align='right'></h1>

Self Organizing Maps
--------------------

MiniSom is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Networks able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

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
print "Training..."
som.train_random(data, 100) # trains the SOM with 100 iterations
print "...ready!"
```

MiniSom implements two types of training. The random training (implemented by the method `train_random`), where the model is trained picking random samples from your data, and the batch training (implemented by the method `train_batch`), where the samples are picked in the order they are stored.

A data driven initialization of the weights is also provided by the method `random_weights_init` which initializes the weights picking random samples from the data.

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

Examples
---------------------
In the directory `examples` you will find the code to produce the following images.

1. Iris flower <a href="http://en.wikipedia.org/wiki/Iris_flower_data_set">dataset</a>.

<img src="http://1.bp.blogspot.com/-j6L__LOB-UI/Ud7BXLLonBI/AAAAAAAAAqU/yf7RYfAoGWM/s1600/iris.png" height="312" width="412" alt="Iris example">

For each observation we have a marker placed on the position of the winning neuron on the map. Each type of marker represents a class of the iris data. The average distance map of the weights is used as background (see the color bar on the right to associate the value) 

2. Images clustering

<img src="http://1.bp.blogspot.com/-DOfulhSC7b8/UjHgeP6oasI/AAAAAAAAAso/t1cChUJZpVg/s1600/digits_mrk.png" height="312" width="412" alt="handwritten digitts recognition">

The graph above represent each image with the handwritten digit it contains. The position corresponds to the position of the winning neuron for the image. Here we also have a version of this graphs that shows the original images:

<img src="http://1.bp.blogspot.com/-VxpdlXkeXfc/UjHgePQIvuI/AAAAAAAAAss/1jOaJRswqzM/s1600/digits_imgs.png" height="312" width="412" alt="handwritten digitts recognition">

3. Color quantization

<img src="http://2.bp.blogspot.com/--b04KEYZPyo/UepdhilpH2I/AAAAAAAAAq4/TefYKHi_uZ8/s1600/qnt_res.png" height="312" width="412" alt="Color quantization example">

4. Natural language processing

<img src="https://3.bp.blogspot.com/-D-z_xeWMHuU/WZb7Kc0fK9I/AAAAAAAABKY/xdI_ApUZMx8O4uN9ihZ_e6jbmQYhUggqgCLcBGAs/s1600/poems_som.png" height="650" width="650">

In this example each poem is associate with a cell in the map. The color represent the author. Check out the notebook in the examples for more details: https://github.com/JustGlowing/minisom/blob/master/examples/PoemsAnalysis.ipynb

Who uses Minisom?
------------
<ul>
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
Minisom has been tested under python 2.7.3 and 3.2.3.

License
---------------------

MiniSom by Giuseppe Vettigli is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
