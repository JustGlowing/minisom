from .sklearn import MiniSOM
from minisom import MiniSom, _build_iteration_indexes, fast_norm
from numpy import (array, linalg, random, subtract, max,
                   exp, zeros, ones, arange, mean, nan,
                   sqrt, argmin, array_equal,)
from numpy.linalg import norm
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import os
import pickle
import unittest


# minisom.MiniSOM
class TestMinisom(unittest.TestCase):
    def setUp(self):
        self.som = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(self.som._weights[i, j]))
        self.som._weights = zeros((5, 5, 1))  # fake weights
        self.som._weights[2, 3] = 5.0
        self.som._weights[1, 1] = 2.0
        self.hex_som = MiniSom(5, 5, 1, topology='hexagonal')
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(
                    self.hex_som._weights[i, j]))
        self.hex_som._weights = zeros((5, 5, 1))  # fake weights

    def test_inverse_decay_to_zero_function(self):
        C = 3 / 100
        assert self.som._inverse_decay_to_zero(1, 2, 3) == 1 * C / (C + 2)

    def test_linear_decay_to_zero_function(self):
        assert self.som._linear_decay_to_zero(1, 2, 3) == 1 * (1 - 2 / 3)

    def test_inverse_decay_to_one_function(self):
        C = (1 - 1) / 3
        assert self.som._inverse_decay_to_one(1, 2, 3) == 1 / (1 + (2 * C))

    def test_linear_decay_to_one_function(self):
        assert self.som._linear_decay_to_one(1, 2, 3) == 1 + (2 * (1 - 1) / 3)

    def test_asymptotic_decay_function(self):
        assert self.som._asymptotic_decay(1, 2, 3) == 1 / (1 + 2 / (3 / 2))

    def test_fast_norm(self):
        assert fast_norm(array([1, 3])) == sqrt(1+9)

    def test_euclidean_distance(self):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = self.som._euclidean_distance(x, w)
        assert_array_almost_equal(d, [[1.41421356, 1.41421356],
                                      [1.41421356, 1.41421356]])

    def test_cosine_distance(self):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = self.som._cosine_distance(x, w)
        assert_array_almost_equal(d, [[1., 1.],
                                      [1., 1.]])

    def test_manhattan_distance(self):
        x = zeros((1, 2))
        w = ones((2, 2, 2))
        d = self.som._manhattan_distance(x, w)
        assert_array_almost_equal(d, [[2., 2.],
                                      [2., 2.]])

    def test_chebyshev_distance(self):
        x = array([1, 3])
        w = ones((2, 2, 2))
        d = self.som._chebyshev_distance(x, w)
        assert_array_almost_equal(d, [[2., 2.],
                                      [2., 2.]])

    def test_check_input_len(self):
        with self.assertRaises(ValueError):
            self.som.train_batch([[1, 2]], 1)

        with self.assertRaises(ValueError):
            self.som.random_weights_init(array([[1, 2]]))

        with self.assertRaises(ValueError):
            self.som._check_input_len(array([[1, 2]]))

        self.som._check_input_len(array([[1]]))
        self.som._check_input_len([[1]])

    def test_unavailable_neigh_function(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_unavailable_distance_function(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 1, activation_distance='ridethewave')

    def test_gaussian(self):
        bell = self.som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_mexican_hat(self):
        bell = self.som._mexican_hat((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_bubble(self):
        bubble = self.som._bubble((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_triangle(self):
        bubble = self.som._triangle((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_win_map(self):
        winners = self.som.win_map([[5.0], [2.0]])
        assert winners[(2, 3)][0] == [5.0]
        assert winners[(1, 1)][0] == [2.0]

    def test_win_map_indices(self):
        winners = self.som.win_map([[5.0], [2.0]], return_indices=True)
        assert winners[(2, 3)] == [0]
        assert winners[(1, 1)] == [1]

    def test_labels_map(self):
        labels_map = self.som.labels_map([[5.0], [2.0]], ['a', 'b'])
        assert labels_map[(2, 3)]['a'] == 1
        assert labels_map[(1, 1)]['b'] == 1
        with self.assertRaises(ValueError):
            self.som.labels_map([[5.0]], ['a', 'b'])

    def test_activation_reponse(self):
        response = self.som.activation_response([[5.0], [2.0]])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_distance_from_weights(self):
        data = arange(-5, 5).reshape(-1, 1)
        weights = self.som._weights.reshape(-1, self.som._weights.shape[2])
        distances = self.som._distance_from_weights(data)
        for i in range(len(data)):
            for j in range(len(weights)):
                assert (distances[i][j] == norm(data[i] - weights[j]))

    def test_quantization_error(self):
        assert self.som.quantization_error([[5], [2]]) == 0.0
        assert self.som.quantization_error([[4], [1]]) == 1.0

    def test_topographic_error(self):
        # 5 will have bmu_1 in (2,3) and bmu_2 in (2, 4)
        # which are in the same neighborhood
        self.som._weights[2, 4] = 6.0
        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        self.som._weights[4, 4] = 15.0
        self.som._weights[0, 0] = 14.
        assert self.som.topographic_error([[5]]) == 0.0
        assert self.som.topographic_error([[15]]) == 1.0

    def test_hexagonal_topographic_error(self):
        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        self.hex_som._weights[4, 4] = 15.0
        self.hex_som._weights[0, 0] = 14.

        # 10 bmu_1 and bmu_2 of 10 are in the same neighborhood
        self.hex_som._weights[2, 2] = 10.0
        self.hex_som._weights[2, 3] = 9.0

        assert self.hex_som.topographic_error([[10]]) == 0.0
        assert self.hex_som.topographic_error([[15]]) == 1.0

    def test_quantization(self):
        q = self.som.quantization(array([[4], [2]]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_divergence_measure(self):
        # test that doesn't use vectorization
        test_data = array([[4], [2]])
        r = 0
        for d in test_data:
            for i in self.som._neigx:
                for j in self.som._neigy:
                    w = self.som.get_weights()[i, j]
                    h = self.som.neighborhood(self.som.winner(d),
                                              self.som._sigma)[i, j]
                    r += h * norm(d - w)
        assert_array_almost_equal(r, self.som.divergence_measure(test_data))

        # handwritten test
        som = MiniSom(2, 1, 2, random_seed=1)
        som._weights = array([[[0.,  1.]], [[1., 0.]]])
        test_data = array([[1., 0.], [0., 1.]])
        h1 = som.neighborhood(som.winner(test_data[0]), som._sigma)
        h2 = som.neighborhood(som.winner(test_data[1]), som._sigma)
        r = h1[0][0] * sqrt(2) + h1[1][0] * 0
        r += h2[0][0] * 0 + h2[1][0] * sqrt(2)
        assert_array_almost_equal(r, som.divergence_measure(test_data))

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        assert_array_almost_equal(som1._weights, som2._weights)
        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10)
        assert q1 > som.quantization_error(data)

        data = array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10, random_order=True)
        assert q1 > som.quantization_error(data)

        data = array([[1, 5], [6, 7]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_use_epochs(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train(data, 10, use_epochs=True)
        assert q1 > som.quantization_error(data)

    def test_train_fixed_points(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        som.train(data, 10, fixed_points={0: (0, 0)})
        with self.assertRaises(ValueError):
            som.train(data, 10, fixed_points={0: (5, 0)})
        with self.assertRaises(ValueError):
            som.train(data, 10, fixed_points={2: (0, 0)})
        with self.assertRaises(ValueError):
            som.train(data, 10, fixed_points={0: (-1, 0)})
        with self.assertRaises(ValueError):
            som.train(data, 10, fixed_points={-1: (0, 0)})
        with self.assertRaises(TypeError):
            som.train(data, 10, fixed_points={'oops': (0, 0)})

    def test_use_epochs_variables(self):
        len_data = 100000
        num_epochs = 100
        random_gen = random.RandomState(1)
        iterations = _build_iteration_indexes(len_data, num_epochs,
                                              random_generator=random_gen,
                                              use_epochs=True)
        assert num_epochs*len_data == len(iterations)

        # checks whether all epochs share the same order of indexes
        first_epoch = iterations[0:len_data]
        for i in range(num_epochs):
            i_epoch = iterations[i*len_data:(i+1)*len_data]
            assert array_equal(first_epoch, i_epoch)

        # checks whether the decay_factor stays constant during one epoch
        # and that its values range from 0 to num_epochs-1
        decay_factors = []
        for t, iteration in enumerate(iterations):
            decay_factor = int(t / len_data)
            decay_factors.append(decay_factor)
        for i in range(num_epochs):
            decay_factors_i_epoch = decay_factors[i*len_data:(i+1)*len_data]
            assert decay_factors_i_epoch == [i]*len_data

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
        for w in som._weights:
            assert_array_equal(w[0], array([1.0, .0]))

    def test_pca_weights_init(self):
        som = MiniSom(2, 2, 2)
        som.pca_weights_init(array([[1.,  0.], [0., 1.], [1., 0.], [0., 1.]]))
        expected = array([[[0.21132487, -1.78867513],
                           [1.78867513, -0.21132487]],
                          [[-1.78867513, 0.21132487],
                           [-0.21132487, 1.78867513]]])
        assert_array_almost_equal(som._weights, expected)

    def test_distance_map(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som._weights = array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        assert_array_equal(som.distance_map(), array([[1., 1.], [1., 1.]]))

        som = MiniSom(2, 2, 2, topology='hexagonal', random_seed=1)
        som._weights = array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        assert_array_equal(som.distance_map(), array([[.5, 1.], [1., .5]]))

        som = MiniSom(3, 3, 1, random_seed=1)
        som._weights = array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        dist = array([[2/3, 3/5, 2/3], [3/5, 4/8, 3/5], [2/3, 3/5, 2/3]])
        assert_array_equal(som.distance_map(scaling='mean'), dist/max(dist))

        with self.assertRaises(ValueError):
            som.distance_map(scaling='puppies')

    def test_pickling(self):
        with open('som.p', 'wb') as outfile:
            pickle.dump(self.som, outfile)
        with open('som.p', 'rb') as infile:
            pickle.load(infile)
        os.remove('som.p')

    def test_callable_activation_distance(self):
        def euclidean(x, w):
            return linalg.norm(subtract(x, w), axis=-1)

        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5,
                       activation_distance=euclidean, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)

    def test_decay_function_value(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 2, decay_function='strawberry')
        MiniSom(5, 5, 2, decay_function='linear_decay_to_zero')
        som1 = MiniSom(5, 5, 2, decay_function=lambda x, y, z: 1)
        som1.train(random.rand(100, 2), 10)

    def test_sigma_decay_function_value(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 2, sigma_decay_function='strawberry')
        som1 = MiniSom(5, 5, 2, sigma_decay_function='linear_decay_to_one')
        som1.train(random.rand(100, 2), 10)


# sklearn.MiniSOM
class TestMiniSOM(unittest.TestCase):
    def test_initialization(self):
        som = MiniSOM(x=2, y=10, sigma=2.0, learning_rate=0.7,
                      num_iteration=100, neighborhood_function='test1',
                      topology='test2', activation_distance='test3')

        self.assertEqual(som.x, 2)
        self.assertEqual(som.y, 10)
        self.assertEqual(som.sigma, 2.0)
        self.assertEqual(som.learning_rate, 0.7)
        self.assertEqual(som.num_iteration, 100)
        self.assertEqual(som.neighborhood_function, 'test1')
        self.assertEqual(som.topology, 'test2')
        self.assertEqual(som.activation_distance, 'test3')

    def test_fit(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)

        self.assertIsNotNone(som.som)

    def test_fit(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)

        self.assertIsNotNone(som.som)

    def test_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X)
        transformed = som.transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        som.fit(X, y=[1, 2, 3, 4])
        predicted = som.predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_transform(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        transformed = som.fit_transform(X)

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_transform_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)

        transformed = som.fit_transform(X)
        transformed_with_y = som.fit_transform(X, y=[1, 2, 3, 4])

        self.assertEqual(transformed.shape, (X.shape[0], 2))

    def test_fit_predict(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000, random_seed=42)

        predicted = som.fit_predict(X)

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_fit_predict_set_y(self):
        X = array([[1, 2], [3, 4], [5, 6], [7, 8]])
        som = MiniSOM(x=3, y=3, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000, random_seed=42)

        predicted = som.fit_predict(X)
        predicted_with_y = som.fit_predict(X, y=[1, 2, 3, 4])

        self.assertTrue(all(predicted == predicted.astype(int)))
        self.assertEqual(predicted.any(), predicted_with_y.any())
        self.assertEqual(predicted.shape, (X.shape[0],))

    def test_set_params(self):
        som = MiniSOM(x=5, y=5, sigma=1.0, learning_rate=0.5,
                      num_iteration=1000)
        som.set_params(sigma=0.8, learning_rate=0.3)
        self.assertEqual(som.sigma, 0.8)
        self.assertEqual(som.learning_rate, 0.3)
