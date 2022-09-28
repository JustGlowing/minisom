[![Python package](https://github.com/JustGlowing/minisom/actions/workflows/python-package.yml/badge.svg?branch=master&event=push)](https://github.com/JustGlowing/minisom/actions/workflows/python-package.yml)

<h1>MiniSom<img src='https://3.bp.blogspot.com/-_6UDGEHzIrs/WSfiyjmoeRI/AAAAAAAABHw/3UQylcCBEhUfHNhf56WSHBBmQ6g_lXQhwCLcB/s320/minisom_logo.png' align='right'></h1>

Self Organizing Maps
--------------------

MiniSom is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.

Updates about MiniSom are posted on <a href="https://twitter.com/JustGlowing">Twitter</a>.

Installation
---------------------

Just use pip:

    pip install minisom

or download MiniSom to a directory of your choice and use the setup script:

    git clone https://github.com/JustGlowing/minisom.git
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

 Then you can train MiniSom just as follows:

```python
from minisom import MiniSom    
som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100) # trains the SOM with 100 iterations
```

You can obtain the position of the winning neuron on the map for a given sample as follows:

```
som.winner(data[0])
```

For an overview of all the features implemented in minisom you can browse the following examples: https://github.com/JustGlowing/minisom/tree/master/examples

#### Export a SOM and load it again

A model can be saved using pickle as follows

```python
import pickle
som = MiniSom(7, 7, 4)

# ...train the som here

# saving the som in the file som.p
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
```

and can be loaded as follows

```python
with open('som.p', 'rb') as infile:
    som = pickle.load(infile)
```

Note that if a lambda function is used to define the decay factor MiniSom will not be pickable anymore.

Explore parameters
---------------------

You can use this dashboard to explore the effect of the parameters on a sample dataset: https://share.streamlit.io/justglowing/minisom/dashboard/dashboard.py

Examples
---------------------

Here are some of the charts you'll see how to generate in the [examples](https://github.com/JustGlowing/minisom/tree/master/examples):

| | |
:-------------------------:|:-------------------------:
Seeds map ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed.png)  | Class assignment ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed_pies.png)
Handwritteng digits mapping ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_digts.png)  |  Hexagonal Topology <img src="https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_seed_hex.png" alt="som hexagonal toplogy" width=450>
Color quantization ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_color_quantization.png)  |  Outliers detection ![](https://github.com/JustGlowing/minisom/raw/master/examples/resulting_images/som_outliers_detection_circle.png)

Other tutorials
------------
- <a href="https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html">Self Organizing Maps on the Glowing Python</a>
- <a href="http://aa.ssdi.di.fct.unl.pt/files/AA-16_notes.pdf">Lecture notes from the Machine Learning course at the University of Lisbon</a>
- <a href="https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d">Introduction to Self-Organizing</a> by Derrick Mwiti
- <a href="http://inphronesys.com/?p=625">Self Organizing Maps on gapminder data</a> [in German]
- <a href="https://medium.com/neuronio/discovering-som-an-unsupervised-neural-network-12e787f38f9">Discovering SOM, an Unsupervised Neural Network</a> by Gisely Alves
- Video tutorials made by the GeoEngineerings School: <a href="https://www.youtube.com/watch?v=3osKNPyAxPM&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=11">Part 1</a>; <a href="https://www.youtube.com/watch?v=uUpQ6MITlVs&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=12">Part 2</a>; <a href="https://www.youtube.com/watch?v=mryFU0TEInk&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC&index=13">Part 3</a>; <a href="https://www.youtube.com/watch?v=9MzFOIoxxdk&index=14&list=PL-i8do33HJovC7xFKaYO21qT37vORJWXC">Part 4</a>
- Video tutorial <a href="https://www.youtube.com/watch?v=0qtvb_Nx2tA">Self Organizing Maps: Introduction</a> by Art of Visualization
- Video tutorial <a href="https://www.youtube.com/watch?v=O6nzwAc_hrQ">Self Organizing Maps Hyperparameter tuning</a> by 
SuperDataScience Machine Learning
- <a href="http://docs.unigrafia.fi/publications/kohonen_teuvo/">MATLAB Implementations and Applications of the Self-Organizing Map</a> by Teuvo Kohonen (Inventor of SOM)

How to cite MiniSom
------------
```
@misc{vettigliminisom,
  title={MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map},
  author={Giuseppe Vettigli},
  year={2018},
  url={https://github.com/JustGlowing/minisom/},
}
```

MiniSom has been cited more than 100 times, check out the research where MiniSom was used  <a href="https://scholar.google.co.uk/scholar?hl=en&as_sdt=2007&q=%22JustGlowing%2Fminisom%22+OR+%22minisom+library%22+OR+%22minisom+python%22+OR+%22minisom+vettigli%22&btnG=">here</a>.

Guidelines to contribute
---------------------
1. In the description of your Pull Request explain clearly what does it implements/fixes and your changes. Possibly give an example in the description of the PR. In cases that the PR is about a code speedup, report a reproducible example and quantify the speedup.
2. Give your pull request a helpful title that summarises what your contribution does. 
3. Write unit tests for your code and make sure the existing tests are up to date. `pytest` can be used for this:
```
pytest minisom.py
```
4. Make sure that there a no stylistic issues using `pycodestyle`:
```
pycodestyle minisom.py
```
5. Make sure your code is properly commented and documented. Each public method needs to be documented as the existing ones.

