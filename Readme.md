MiniSom
====================

![MiniSom]( http://1.bp.blogspot.com/-tD8Kg6FEOcg/Uc_wjGZ7qaI/AAAAAAAAAo4/A4Q1_dbqVLo/s278/logo.png "MiniSom")

Self Organizing Maps
--------------------

MiniSom is minimalistic implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Networks able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.

Installation
---------------------

    python setup.py install

How to use it
---------------------

In order to use MiniSom you need your data organized as a Numpy matrix where each row corresponds to an observation or an as list of lists like the following:

	data = [[ 5.1  3.5  1.4  0.2],
	        [ 4.9  3.   1.4  0.2],
	        [ 4.7  3.2  1.3  0.2], # <-- single pattern
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

#### Result interpretation and visualization

After the training, the method `winner(x)` computes the coordinates of the mapping of `x` on the map and it can be used to plot:



### Other features

MiniSom implements two types of training. The random training (implemente by the method `train_random`), where the model is trained picking random samples from your data, and the batch training (implemente by the method `train_batch`), where the samples are used in the order they are stored.

MiniSom initializes the neurons weights at random. A data driven initialization is also provided by the method `random_weights_init` which initializes the weights picking random samples from the data.

Planned improvements
---------------------
*Implement a classification method.

License
---------------------

MiniSom by Giuseppe Vettigli is licensed under the Creative Commons Attribution 3.0 Unported License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/ "http://creativecommons.org/licenses/by/3.0/").

![License]( http://i.creativecommons.org/l/by/3.0/88x31.png "Creative Commons Attribution 3.0 Unported License")
