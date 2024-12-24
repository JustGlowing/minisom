from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal, isclose)
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os

# for unit tests

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    if use_epochs:
        iterations_per_epoch = arange(data_len)
        if random_generator:
            random_generator.shuffle(iterations_per_epoch)
        iterations = tile(iterations_per_epoch, num_iterations)
    else:
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))


class MiniSom(object):
    Y_HEX_CONV_FACTOR = (3.0 / 2.0) / sqrt(3)

    def __init__(self, x, y, input_len, sigma=1, learning_rate=0.5,
                 decay_function='asymptotic_decay',
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None,
                 sigma_decay_function='asymptotic_decay'):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1)
            Spread of the neighborhood function.

            Needs to be adequate to the dimensions of the map
            and the neighborhood function. In some cases it
            helps to set sigma as sqrt(x^2 +y^2).

        learning_rate : float, optional (default=0.5)
            Initial learning rate.

            Adequate values are dependent on the data used for training.

            By default, at the iteration t, we have:
                learning_rate(t) = learning_rate / (1 + t * (100 / max_iter))

        decay_function : string or callable, optional
        (default='inverse_decay_to_zero')
            Function that reduces learning_rate at each iteration.
            Possible values: 'inverse_decay_to_zero', 'linear_decay_to_zero',
                             'asymptotic_decay' or callable

            If a custom decay function using a callable
            it will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed

            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1)

        random_seed : int, optional (default=None)
            Random seed to use.

        sigma_decay_function : string, optional
        (default='inverse_decay_to_one')
            Function that reduces sigma at each iteration.
            Possible values: 'inverse_decay_to_one', 'linear_decay_to_one',
                             'asymptotic_decay'

            The default function is:
                sigma(t) = sigma / (1 + (t * (sigma - 1) / max_iter))
        """
        if sigma > sqrt(x*x + y*y):
            warn('Warning: sigma might be too high ' +
                 'for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            self._yy *= self.Y_HEX_CONV_FACTOR
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        lr_decay_functions = {
            'inverse_decay_to_zero': self._inverse_decay_to_zero,
            'linear_decay_to_zero': self._linear_decay_to_zero,
            'asymptotic_decay': self._asymptotic_decay}

        if isinstance(decay_function, str):
            if decay_function not in lr_decay_functions:
                msg = '%s not supported. Functions available: %s'
                raise ValueError(msg % (decay_function,
                                        ', '.join(lr_decay_functions.keys())))

            self._learning_rate_decay_function = \
                lr_decay_functions[decay_function]
        elif callable(decay_function):
            self._learning_rate_decay_function = decay_function

        sig_decay_functions = {
            'inverse_decay_to_one': self._inverse_decay_to_one,
            'linear_decay_to_one': self._linear_decay_to_one,
            'asymptotic_decay': self._asymptotic_decay}

        if sigma_decay_function not in sig_decay_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (sigma_decay_function,
                                    ', '.join(sig_decay_functions.keys())))

        self._sigma_decay_function = sig_decay_functions[sigma_decay_function]

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T, self._yy.T

    def convert_map_to_euclidean(self, xy):
        """Converts map coordinates into euclidean coordinates
        that reflects the chosen topology.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T[xy], self._yy.T[xy]

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _inverse_decay_to_zero(self, learning_rate, t, max_iter):
        """Decay function of the learning process that asymptotically
        approaches zero.
        """
        C = max_iter / 100.0
        return learning_rate * C / (C + t)

    def _linear_decay_to_zero(self, learning_rate, t, max_iter):
        """Decay function of the learning process that linearly
        decreases to zero.
        """
        return learning_rate * (1 - t / max_iter)

    def _inverse_decay_to_one(self, sigma, t, max_iter):
        """Decay function of sigma that asymptotically approaches one.
        """
        C = (sigma - 1) / max_iter
        return sigma / (1 + (t * C))

    def _linear_decay_to_one(self, sigma, t, max_iter):
        """Decay function of sigma that linearly decreases
        to one.
        """
        return sigma + (t * (1 - sigma) / max_iter)

    def _asymptotic_decay(self, dynamic_parameter, t, max_iter):
        """Decay function of the learning process
        and sigma that decays these values asymptotically to 1/3
        of their original values.
        """
        return dynamic_parameter / (1 + t / (max_iter / 2))

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        p = power(self._xx-self._xx.T[c], 2) + power(self._yy-self._yy.T[c], 2)
        d = 2*sigma*sigma
        return (exp(-p/d)*(1-2/d*p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            rate of decay for sigma and learning rate
        max_iteration : int
            If use_epochs is True:
                Number of epochs the SOM will be trained for
            If use_epochs is False:
                Maximum number of iterations (one iteration per sample).
        """
        eta = self._learning_rate_decay_function(self._learning_rate,
                                                 t, max_iteration)
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, eigvecs = linalg.eig(cov(data))
        pc = (eigvecs.T @ data)
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1*pc[pc_order[0]] + \
                                      c2*pc[pc_order[1]]

    def _check_fixed_points(self, fixed_points, data):
        for k in fixed_points.keys():
            if not isinstance(k, int):
                raise TypeError(f'fixed points indexes must ' +
                                'be integers.')
            if k >= len(data) or k < 0:
                raise ValueError(f'an index of a fixed point ' +
                                 'cannot be grater than len(data)' +
                                 ' or less than 0.')
            if fixed_points[k][0] >= self._weights.shape[0] or \
               fixed_points[k][1] >= self._weights.shape[1]:
                raise ValueError(f'coordinates for fixed point' +
                                 ' are out of boundaries.')
            if fixed_points[k][0] < 0 or \
               fixed_points[k][1] < 0:
                raise ValueError(f'coordinates cannot be negative.')

    def train(self, data, num_iteration,
              random_order=False, verbose=False,
              use_epochs=False, fixed_points=None):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            If use_epochs is False, the weights will be
            updated num_iteration times. Otherwise they will be updated
            len(data)*num_iteration times.

        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        verbose : bool (default=False)
            If True the status of the training will be
            printed each time the weights are updated.

        use_epochs : bool (default=False)
            If True the SOM will be trained for num_iteration epochs.
            In one epoch the weights are updated len(data) times and
            the learning rate is constat throughout a single epoch.

        fixed_points : dict (default=None)
            A dictionary k : (c_1, c_2), that will force the
            training algorithm to use the neuron with coordinates
            (c_1, c_2) as winner for the sample k instead of
            the best matching unit.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator,
                                              use_epochs)
        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)

        if fixed_points:
            self._check_fixed_points(fixed_points, data)
        else:
            fixed_points = {}

        for t, iteration in enumerate(iterations):
            decay_rate = get_decay_rate(t, len(data))
            self.update(data[iteration],
                        fixed_points.get(iteration,
                                         self.winner(data[iteration])),
                        decay_rate, num_iteration)
        if verbose:
            print('\n quantization error:', self.quantization_error(data))

    def train_random(self, data, num_iteration, verbose=False):
        """Trains the SOM picking samples at random from data.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each time the weights are updated.
        """
        self.train(data, num_iteration, random_order=True, verbose=verbose)

    def train_batch(self, data, num_iteration, verbose=False):
        """Trains the SOM using all the vectors in data sequentially.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each time the weights are updated.
        """
        self.train(data, num_iteration, random_order=False, verbose=verbose)

    def distance_map(self, scaling='sum'):
        """Returns the distance map of the weights.
        If scaling is 'sum' (default), each cell is the normalised sum of
        the distances between a neuron and its neighbours. Note that this
        method uses the euclidean distance.

        Parameters
        ----------
        scaling : string (default='sum')
            If set to 'mean', each cell will be the normalized
            by the average of the distances of the neighbours.
            If set to 'sum', the normalization is done
            by the sum of the distances.
        """

        if scaling not in ['sum', 'mean']:
            raise ValueError(f'scaling should be either "sum" or "mean" ('
                             f'"{scaling}" not valid)')

        um = nan * zeros((self._weights.shape[0],
                          self._weights.shape[1],
                          8))  # 2 spots more for hexagonal topology

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

        if self.topology == 'hexagonal':
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                w_2 = self._weights[x, y]
                e = y % 2 == 0   # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < self._weights.shape[0] and
                            y+j >= 0 and y+j < self._weights.shape[1]):
                        w_1 = self._weights[x+i, y+j]
                        um[x, y, k] = fast_norm(w_2-w_1)

        if scaling == 'mean':
            um = nanmean(um, axis=2)
        if scaling == 'sum':
            um = nansum(um, axis=2)

        return um/um.max()

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return norm(data-self.quantization(data), axis=1).mean()

    def divergence_measure(self, data):
        """Returns the divergence measure computed as
           sum_i, sum_c (neighborhood(c, sigma) * || d_i - w_c ||^2
        """
        divergence = 0
        for d in data:
            divergence += multiply(self.neighborhood(self.winner(d),
                                                     self._sigma),
                                   norm(d - self.get_weights(), axis=2)).sum()
        return divergence

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = prod(self._activation_map.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan
        if self.topology == 'hexagonal':
            return self._topographic_error_hexagonal(data)
        else:
            return self._topographic_error_rectangular(data)

    def _topographic_error_hexagonal(self, data):
        """Return the topographic error for hexagonal grid"""
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2mu_coords = [[self._get_euclidean_coordinates_from_index(bmu[0]),
                        self._get_euclidean_coordinates_from_index(bmu[1])]
                       for bmu in b2mu_inds]
        b2mu_coords = array(b2mu_coords)
        b2mu_neighbors = [isclose(1, norm(bmu1 - bmu2))
                          for bmu1, bmu2 in b2mu_coords]
        te = 1 - mean(b2mu_neighbors)
        return te

    def _topographic_error_rectangular(self, data):
        """Return the topographic error for rectangular grid"""
        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2my_xy = unravel_index(b2mu_inds, self._weights.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = hstack([diff(b2mu_x), diff(b2mu_y)])
        distance = norm(dxdy, axis=1)
        return (distance > t).mean()

    def _get_euclidean_coordinates_from_index(self, index):
        """Returns the Euclidean coordinated of a neuron using its
        index as the input"""
        if index < 0:
            return (-1, -1)
        y = self._weights.shape[1]
        coords = self.convert_map_to_euclidean((int(index/y), index % y))
        return coords

    def win_map(self, data, return_indices=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.winner(x)].append(i if return_indices else x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap
