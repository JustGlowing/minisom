from numpy import meshgrid,sqrt,sqrt,array,unravel_index,nditer,linalg,random,subtract,power,exp,pi,zeros

"""
    Minimalistic implementation of the Self Organizing Maps (SOM)

    Giuseppe Vettigli 2013.
"""

class MiniSom:
    def __init__(self,x,y,input_len,sigma=0.1,learning_rate=0.5):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where is #num_iteration/2)
        """
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = random.rand(x,y,input_len)*2-1 # random initialization
        self.activation_map = zeros((x,y))
        self.neigx,self.neigy = meshgrid(range(y),range(x)) # used to evaluate the neighborhood function    

    def _activate(self,x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = subtract(x,self.weights) # x - w
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = linalg.norm(s[it.multi_index]) # || x - w ||
            it.iternext()

    def activate(self,x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def gaussian(self,c,sigma=0.1):
        """ Bidimentional Gaussian centered in c """
        d = sqrt( power((c[0]-self.neigx),2) + power((c[1]-self.neigy),2) )
        return exp(-(d*d))/(2*pi*sigma) # a matrix is returned

    def winner(self,x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(),self.activation_map.shape)

    def update(self,x,win,t):
        """
            Updates the weights of the neurons.
            x - current pattern to learning
            win - position of the winning neuron for x (array or tuple).
            eta - learning rate
            t - iteration index
        """
        # eta(t) = eta(0) / (1 + t/T) 
        # keeps the learning rate nearly constant for the first T iterations and then adjusts it
        eta = self.learning_rate/(1+t/self.T)
        g = self.gaussian(win,self.sigma)*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])            
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self,data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i,x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q


    def random_weights_init(self,data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[int(random.rand()*len(data)-1)]
            self.weights[it.multi_index] = self.weights[it.multi_index]/linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self,data,num_iteration):        
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)        
        for iteration in range(num_iteration):
            rand_i = int(round(random.rand()*len(data)-1)) # pick a random sample          
            self.update(data[rand_i],self.winner(data[rand_i]),iteration)

    def train_batch(self,data,num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx],self.winner(data[idx]),iteration)
            iteration += 1

    def _init_T(self,num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2 # keeps the learning rate nearly constant for the first half of the iterations

    def distance_map(self):
        """ Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1) """
        um = zeros((self.weights.shape[0],self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1,it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1,it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += linalg.norm(self.weights[ii,jj,:]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self,data):
        """ 
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self.weights.shape[0],self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a
    