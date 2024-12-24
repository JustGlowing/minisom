from minisom import MiniSom
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from numpy import array, ravel_multi_index, all

# for unit tests
import unittest


class MiniSOM(BaseEstimator, TransformerMixin):
    def __init__(self, x=10, y=10, sigma=1.0, learning_rate=0.5,
                 num_iteration=1000,
                 decay_function='asymptotic_decay',
                 neighborhood_function='gaussian',
                 topology='rectangular',
                 activation_distance='euclidean', random_seed=None,
                 sigma_decay_function='asymptotic_decay',
                 random_order=False, verbose=False,
                 use_epochs=False, fixed_points=None):
        """
        MiniSOM wrapper that integrates seamlessly with the
        Scikit-learn ecosystem, providing a familiar API for users.

        It enables easy integration with Scikit-learn pipelines and
        tools like GridSearchCV for hyperparameter optimization.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        sigma : float, optional (default=1)
            Spread of the neighborhood function.

            Needs to be adequate to the dimensions of the map
            and the neighborhood function. In some cases it
            helps to set sigma as sqrt(x^2 +y^2).

        learning_rate : float, optional (default=0.5)
            Initial learning rate.

            Adequate values are dependent on the data used for
            training.

            By default, at the iteration t, we have:
                learning_rate(t) = learning_rate / (1 + t * (100 / max_iter))

        num_iteration : int, optional (default=1000)
            Number of iterations.

            Adequate values are dependent on the data used for training.

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

        random_order : bool (default=False)
            If True, samples in SOM train function are picked in random order.
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

        self.x = x
        self.y = y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_seed = random_seed
        self.sigma_decay_function = sigma_decay_function
        self.random_order = random_order
        self.verbose = verbose
        self.use_epochs = use_epochs
        self.fixed_points = fixed_points

    def fit(self, X, y=None):
        """
        Initializes SOM algorithm from minisom library
        and fits on it data matrix.

        Parameters
        ----------
        X : np.array or list
            Data matrix.


        y : Ignored
            Not used, present here for API consistency by convention.
        """
        if sparse.issparse(X):
            X = X.toarray()

        self.som = MiniSom(self.x, self.y, X.shape[1],
                           sigma=self.sigma,
                           learning_rate=self.learning_rate,
                           decay_function=self.decay_function,
                           neighborhood_function=self.neighborhood_function,
                           topology=self.topology,
                           activation_distance=self.activation_distance,
                           random_seed=self.random_seed,
                           sigma_decay_function=self.sigma_decay_function
                           )

        self.som.train(X, self.num_iteration,
                       random_order=self.random_order,
                       verbose=self.verbose,
                       use_epochs=self.use_epochs,
                       fixed_points=self.fixed_points
                       )
        return self

    def transform(self, X):
        """
        Transform the data by finding the best matching unit (BMU) for
        each sample. Returns the BMU coordinates for each input sample.

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        if sparse.issparse(X):
            X = X.toarray()
        return array([self.som.winner(x) for x in X])

    def predict(self, X):
        """
        Predict the cluster assignment (BMU) for each sample.
        Here, we return the grid position (BMU) for each sample
        as the cluster assignment.
        We treat each unique BMU as a distinct cluster.

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        bmu_coords = array([self.som.winner(x) for x in X])
        bmu_labels = ravel_multi_index(bmu_coords.T, (self.x, self.y))
        return bmu_labels

    def fit_transform(self, X, y=None):
        """
        Fit the SOM and return the transformed
        BMU coordinates for each input sample.

        Convenience method; equivalent to calling fit(X) followed by
        transform(X).

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.transform(X)

    def fit_predict(self, X, y=None):
        """
        Fit the SOM and return predicted cluster assignments
        (BMU) for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : np.array or list
            Data matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if sparse.issparse(X):
            X = X.toarray()
        self.fit(X)
        return self.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Helpful for hyperparameter tuning using GridSearchCV.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "x": self.x,
            "y": self.y,
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "num_iteration": self.num_iteration,
            "decay_function": self.decay_function,
            "neighborhood_function": self.neighborhood_function,
            "topology": self.topology,
            "activation_distance": self.activation_distance,
            "random_seed": self.random_seed,
            "sigma_decay_function": self.sigma_decay_function,
            "random_order": self.random_order,
            "verbose": self.verbose,
            "use_epochs": self.use_epochs,
            "fixed_points": self.fixed_points
        }

    def set_params(self, **params):

        """
        Set the parameters of this estimator.
        This allows setting parameters during GridSearchCV.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
