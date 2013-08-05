MiniSom
====================

![MiniSom]( http://3.bp.blogspot.com/-TjLGnec3uko/Ud8LbHTpO1I/AAAAAAAAAqk/nfJneFOZrK8/s1600/logo.png "MiniSom")

Self Organizing Maps
--------------------

MiniSom is minimalistic Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Networks able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

Installation
---------------------

    python setup.py install

How to use it
---------------------

In order to use MiniSom you need your data organized as a Numpy matrix where each row corresponds to an observation or an as list of lists like the following:

    data = [[ 5.1  3.5  1.4  0.2],
            [ 4.9  3.   1.4  0.2],
            [ 4.7  3.2  1.3  0.2], # <-- single observation
            [ 4.6  3.1  1.5  0.2],
            [ 5.   3.6  1.4  0.2],
            [ 4.1  3.3  1.4  0.2],
            [ 4.2  3.2  1.2  0.2]]         

 Then you can run MiniSom just as follows:

    from minisom import MiniSom    
    som = MiniSom(6,6,4,sigma=0.3,learning_rate=0.5) # initialization of 6x6 SOM
    print "Training..."
    som.train_random(data,100) # trains the SOM with 100 iterations
    print "...ready!"

### Using the trained SOM

After the training MiniSom makes you able to

* Compute the coordinate assigned to an observation `x` on the map with the method `winner(x)`.
* Compute the average distance map of the weights on the map with the method `distance_map`.
* Compute the number of times that each neuron have been considered winner for the observations of a new data set with the method `activation_response(data)`.

#### Training algorithms

MiniSom implements two types of training. The random training (implemented by the method `train_random`), where the model is trained picking random samples from your data, and the batch training (implemented by the method `train_batch`), where the samples are used in the order they are stored.

#### Weights initialization

MiniSom initializes the neurons weights at random. A data driven initialization is also provided by the method `random_weights_init` which initializes the weights picking random samples from the data.

#### Vector quantization

The data can be quantized by assigning a code book (weights vector of the winning neuron) to each sample in data. This kind of vector quantization is implemented by the method `quantization`.

Examples
---------------------
In examples/example_iris.py you can find an example that shows how to train MiniSom and visualize the result using the <a href="http://en.wikipedia.org/wiki/Iris_flower_data_set">Iris flower dataset</a>. Here is the result of the script:

<img src="http://1.bp.blogspot.com/-j6L__LOB-UI/Ud7BXLLonBI/AAAAAAAAAqU/yf7RYfAoGWM/s1600/iris.png" height="312" width="412" alt="Iris example">

For each winning neuron we have a marker. Each type of marker represents a class of the iris data. The average distance map of the weights is used as background (see the color bar on the right to associate the value). 

In examples/example_color.py you can find an example of how to use MiniSom for color quantization. Here's one of the possible results:

<img src="http://2.bp.blogspot.com/--b04KEYZPyo/UepdhilpH2I/AAAAAAAAAq4/TefYKHi_uZ8/s1600/qnt_res.png" height="312" width="412" alt="Color quantization example">

(the examples require matplotlib for the visualization of the results).

Notes
---------------------
Minisom have been tested under python 2.7.3 and 3.2.3.

License
---------------------

MiniSom by Giuseppe Vettigli is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
