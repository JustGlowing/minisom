import os
import pickle
import pytest
import numpy as np
import numpy.testing as npt
from minisom import MiniSom, _build_iteration_indexes, asymptotic_decay


class TestMinisom():
    
    @pytest.fixture
    def som(self):
        ret = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                npt.assert_almost_equal(1.0, np.linalg.norm(ret._weights[i, j]))
        ret._weights = np.zeros((5, 5, 1))  # fake weights
        ret._weights[2, 3] = 5.0
        ret._weights[1, 1] = 2.0
        return ret

    def test_decay_function(self, som):
        assert som._decay_function(1., 2., 3.) == 1./(1.+2./(3./2))

    def test_euclidean_distance(self, som):
        x = np.zeros((1, 2))
        w = np.ones((2, 2, 2))
        d = som._euclidean_distance(x, w)
        npt.assert_array_almost_equal(d, [[1.41421356, 1.41421356],
                                          [1.41421356, 1.41421356]])

    def test_cosine_distance(self, som):
        x = np.zeros((1, 2))
        w = np.ones((2, 2, 2))
        d = som._cosine_distance(x, w)
        npt.assert_array_almost_equal(d, [[1., 1.],
                                          [1., 1.]])

    def test_manhattan_distance(self, som):
        x = np.zeros((1, 2))
        w = np.ones((2, 2, 2))
        d = som._manhattan_distance(x, w)
        npt.assert_array_almost_equal(d, [[2., 2.],
                                          [2., 2.]])

    def test_chebyshev_distance(self, som):
        x = np.array([1, 3])
        w = np.ones((2, 2, 2))
        d = som._chebyshev_distance(x, w)
        npt.assert_array_almost_equal(d, [[2., 2.],
                                          [2., 2.]])

    def test_check_input_len(self, som):
        with pytest.raises(ValueError):
            som.train_batch([[1, 2]], 1)

        with pytest.raises(ValueError):
            som.random_weights_init(np.array([[1, 2]]))

        with pytest.raises(ValueError):
            som._check_input_len(np.array([[1, 2]]))

        som._check_input_len(np.array([[1]]))
        som._check_input_len([[1]])

    def test_unavailable_neigh_function(self):
        with pytest.raises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_unavailable_distance_function(self):
        with pytest.raises(ValueError):
            MiniSom(5, 5, 1, activation_distance='ridethewave')

    def test_gaussian(self, som):
        bell = som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_mexican_hat(self, som):
        bell = som._mexican_hat((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_bubble(self, som):
        bubble = som._bubble((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_triangle(self, som):
        bubble = som._triangle((2, 2), 1)
        assert bubble[2, 2] == 1
        assert sum(sum(bubble)) == 1

    def test_win_map(self, som):
        winners = som.win_map(np.array([[5.0], [2.0]]))
        assert winners[(2, 3)][0] == 5.0
        assert winners[(1, 1)][0] == 2.0

    def test_win_map_indices(self, som):
        winners = som.win_map([[5.0], [2.0]], return_indices=True)
        assert winners[(2, 3)] == np.array([0])
        assert winners[(1, 1)] == np.array([1])


    def test_labels_map(self, som):
        labels_map = som.labels_map(np.array([[5.0], [2.0]]), ['a', 'b'])
        assert labels_map[(2, 3)]['a'] == 1
        assert labels_map[(1, 1)]['b'] == 1
        with pytest.raises(ValueError):
            som.labels_map([[5.0]], ['a', 'b'])

    def test_activation_reponse(self, som):
        response = som.activation_response(np.array([[5.0], [2.0]]))
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self, som):
        assert som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_distance_from_weights(self, som):
        data = np.arange(-5., 5.).reshape(-1, 1)
        weights = som._weights.reshape(-1, som._weights.shape[2])
        distances = som._distance_from_weights(data)
        for i in range(len(data)):
            for j in range(len(weights)):
                assert (distances[i][j] == np.linalg.norm(data[i] - weights[j]))

    def test_quantization_error(self, som):
        assert som.quantization_error(np.array([[5.], [2.]])) == 0.0
        assert som.quantization_error(np.array([[4.], [1.]])) == 1.0

    def test_topographic_error(self, som):
        # 5 will have bmu_1 in (2,3) and bmu_2 in (2, 4)
        # which are in the same neighborhood
        som._weights[2, 4] = 6.0
        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        som._weights[4, 4] = 15.0
        som._weights[0, 0] = 14.
        assert som.topographic_error(np.array([[5]])) == 0.0
        assert som.topographic_error(np.array([[15]])) == 1.0

    def test_hexagonal_topographic_error(self):
        hex_som = MiniSom(5, 5, 1, topology='hexagonal')
        hex_som._weights = np.zeros((5, 5, 1))

        # 15 will have bmu_1 in (4, 4) and bmu_2 in (0, 0)
        # which are not in the same neighborhood
        hex_som._weights[4, 4] = 15.0
        hex_som._weights[0, 0] = 14.

        # 10 bmu_1 and bmu_2 of 10 are in the same neighborhood
        hex_som._weights[2, 2] = 10.0
        hex_som._weights[2, 3] = 9.0

        assert hex_som.topographic_error(np.array([[10]])) == 0.0
        assert hex_som.topographic_error(np.array([[15]])) == 1.0

    def test_quantization(self, som):
        q = som.quantization(np.array([[4.], [2.]]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        npt.assert_array_almost_equal(som1._weights, som2._weights)
        data = np.random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        npt.assert_array_almost_equal(som1._weights, som2._weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = np.array([[4., 2.], [3., 1.]])
        q1 = som.quantization_error(data)
        som.train(data, 10)
        assert q1 > som.quantization_error(data)

        data = np.array([[1., 5.], [6., 7.]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = np.array([[4., 2.], [3., 1.]])
        q1 = som.quantization_error(data)
        som.train(data, 10, random_order=True)
        assert q1 > som.quantization_error(data)

        data = np.array([[1., 5.], [6., 7.]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10, verbose=True)
        assert q1 > som.quantization_error(data)

    def test_train_use_epochs(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = np.array([[4., 2.], [3., 1.]])
        q1 = som.quantization_error(data)
        som.train(data, 10, use_epochs=True)
        assert q1 > som.quantization_error(data)

    def test_use_epochs_variables(self):
        len_data = 100000
        num_epochs = 100
        random_gen = np.random.RandomState(1)
        iterations = _build_iteration_indexes(len_data, num_epochs,
                                              random_generator=random_gen,
                                              use_epochs=True)
        assert num_epochs*len_data == len(iterations)

        # checks whether all epochs share the same order of indexes
        first_epoch = iterations[0:len_data]
        for i in range(num_epochs):
            i_epoch = iterations[i*len_data:(i+1)*len_data]
            assert np.array_equal(first_epoch, i_epoch)

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
        som.random_weights_init(np.array([[1.0, .0]]))
        for w in som._weights:
            npt.assert_array_equal(w[0], np.array([1.0, .0]))

    def test_pca_weights_init(self):
        som = MiniSom(2, 2, 2)
        som.pca_weights_init(np.array([[1.,  0.], [0., 1.], [1., 0.], [0., 1.]]))
        expected = np.array([[[0.21132487, -1.78867513],
                           [1.78867513, -0.21132487]],
                          [[-1.78867513, 0.21132487],
                           [-0.21132487, 1.78867513]]])
        npt.assert_array_almost_equal(som._weights, expected)

    def test_distance_map(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som._weights = np.array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        npt.assert_array_equal(som.distance_map(), np.array([[1., 1.], [1., 1.]]))

        som = MiniSom(2, 2, 2, topology='hexagonal', random_seed=1)
        som._weights = np.array([[[1.,  0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        npt.assert_array_equal(som.distance_map(), np.array([[.5, 1.], [1., .5]]))

        som = MiniSom(3, 3, 1, random_seed=1)
        som._weights = np.array([
                [[1.], [0.], [1.]],
                [[0.], [1.], [0.]],
                [[1.], [0.], [1.]]
            ])
        dist = np.array([[2/3, 3/5, 2/3], [3/5, 4/8, 3/5], [2/3, 3/5, 2/3]])
        npt.assert_array_equal(som.distance_map(scaling='mean'), dist/dist.max())

        with pytest.raises(ValueError):
            som.distance_map(scaling='puppies')

    def test_pickling(self):
        with open('som.p', 'wb') as outfile:
            pickle.dump(self.som, outfile)
        with open('som.p', 'rb') as infile:
            pickle.load(infile)
        os.remove('som.p')

    def test_callable_activation_distance(self):
        def euclidean(x, w):
            return np.linalg.norm(np.subtract(x, w), axis=-1)

        data = np.random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5,
                       activation_distance=euclidean, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        npt.assert_array_almost_equal(som1._weights, som2._weights)


    def test_asymptotic_decay(self):
        ret = asymptotic_decay(learning_rate=0.05, t=10, max_iter=30)
        npt.assert_array_almost_equal(ret, 0.03)