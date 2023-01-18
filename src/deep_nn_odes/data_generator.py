from abc import ABC, abstractmethod
import numpy as np
import sklearn.datasets


class DataGenerator(ABC):
    """Base class for generating classification data

    Each data point (X_j,y_j) consists of a point X_j in d-dimensional space
    and a class probability y_j in n_c-dimensional space, where n_c is the number of classes

    :arg dim: dimension d of input input space
    :arg n_class; number of classes n_c
    :arg n_train: total number of training samples to generate; must be a multiple of batchsize
    :arg n_test: total number of test samples to generate
    :arg batchsize: batch size
    """

    def __init__(self, dim, n_class, n_train, n_test, batchsize):
        self.dim = dim
        self.n_class = n_class
        assert n_train % batchsize == 0
        self.n_train = n_train
        self.n_test = n_test
        self.batchsize = batchsize
        self.rng = np.random.default_rng(seed=81345357)
        self._Xy_train = np.zeros((self.n_train, self.dim + self.n_class))
        self._Xy_test = np.zeros((self.n_test, self.dim + self.n_class))
        (
            self._Xy_train[:, : self.dim],
            self._Xy_train[:, self.dim :],
        ) = self._generate_data(self.n_train)
        (
            self._Xy_test[:, : self.dim],
            self._Xy_test[:, self.dim :],
        ) = self._generate_data(self.n_test)
        self._generate_data(self.n_test)

    @abstractmethod
    def _generate_data(self, n_data):
        """Generate data points

        This abstract method needs to be implemented in all derived classes

        :arg n_data: number of data points to generate

        Return tuple (X_data,y_data) where X_data is an array of shape (n_data,dim) and
        y_data is an array of shape (n_data,)
        """

    def get_shuffled_batched_train_data(self):
        """Return shuffled and batched training data

        returns tuple (X_train_batched, y_train_batched), where each element is a list
        containing n_train//batchsize arrays of shape (batchsize,dim) and (batchsize,n_c)
        """
        nbatch = self.n_train // self.batchsize
        self.rng.shuffle(self._Xy_train, axis=0)
        X_batched = np.split(self._Xy_train[:, : self.dim], nbatch)
        y_batched = np.split(self._Xy_train[:, self.dim :], nbatch)
        return X_batched, y_batched

    def get_test_data(self):
        """Return the test data

        returns a tuple (X_test, y_test), where the elements are of shape
        (n_test,dim) and (n_test,n_c)
        """
        return self._Xy_test[:, : self.dim], self._Xy_test[:, self.dim :]


class DataGeneratorCircular(DataGenerator):
    """Generator for randomly distributed points in unit square, which are labelled as 1
    if they fall into a circle with radius R around the centre (0.5,0.5) and are labelled
    as 0 if they lie outside this circle.

    :arg n_train: total number of training samples to generate; must be a multiple of batchsize
    :arg n_test: total number of test samples to generate
    :arg batchsize: batch size
    :arg radius: radius R of circle
    """

    def __init__(self, n_train, n_test, batchsize, radius=0.25):
        self.radius = radius
        super().__init__(2, 2, n_train, n_test, batchsize)

    def _generate_data(self, n_data):
        """Generate data points

        Generate n_data points that are uniformly distributed over the unit circle
        and associated labels

        :arg n_data: number of data points to generate

        Return tuple (X_data,y_data) where X_data is an array of shape (n_data,2) and
        y_data is an array of shape (n_data,2)
        """
        X = self.rng.uniform(low=0.0, high=1.0, size=(n_data, 2))
        y_class = np.asarray(
            np.linalg.norm(X - [0.5, 0.5], axis=-1) < self.radius, dtype=int
        )
        # Convert to one-hot encoding
        y_onehot = np.eye(self.n_class)[y_class]
        return X, y_onehot


class DataGeneratorMoons(DataGenerator):
    """Generator for two moons dataset.

    :arg n_train: total number of training samples to generate; must be a multiple of batchsize
    :arg n_test: total number of test samples to generate
    :arg batchsize: batch size
    :arg noise: standard deviation of normal noise
    """

    def __init__(self, n_train, n_test, batchsize, noise=0.2):
        self.noise = noise
        super().__init__(2, 2, n_train, n_test, batchsize)

    def _generate_data(self, n_data):
        """Generate data points

        draw datapoints from two moons dataset

        :arg n_data: number of data points to generate

        Return tuple (X_data,y_data) where X_data is an array of shape (n_data,2) and
        y_data is an array of shape (n_data,2)
        """
        X, y_class = sklearn.datasets.make_moons(n_data, noise=self.noise)
        # Convert to one-hot encoding
        y_onehot = np.eye(self.n_class)[y_class]
        return X, y_onehot


class DataGeneratorSwissRoll(DataGenerator):
    """Generator for swiss roll dataset

    datapoints are drawn from the curves

        f_k(r,theta) = (r+d_k)*(cos(theta),sin(theta)) + (xi_0,xi_1)

    where d_0 = 0, d_1 = 0.2 for the positive/negative samples, 0 <= r <= 1 and
    0 <= theta <= 4*pi. The noise xi_j is drawn from a normal distribution with mean zero
    and the standard variation that is given by the noise parameter that is passed
    to the constructor.

    :arg n_train: total number of training samples to generate; must be a multiple of batchsize
    :arg n_test: total number of test samples to generate
    :arg batchsize: batch size
    :arg noise: standard deviation of normal noise
    """

    def __init__(self, n_train, n_test, batchsize, noise=0.2):
        self.noise = noise
        super().__init__(2, 2, n_train, n_test, batchsize)

    def _generate_data(self, n_data):
        """Generate data points

        draw datapoints from circular dataset

        :arg n_data: number of data points to generate

        Return tuple (X_data,y_data) where X_data is an array of shape (n_data,2) and
        y_data is an array of shape (n_data,2)
        """
        radius = np.linspace(0, 1, n_data // 2)
        circle = np.stack(
            [np.cos(4.0 * np.pi * radius), np.sin(4.0 * np.pi * radius)], axis=-1
        )
        X = np.concatenate(
            [
                np.expand_dims(radius, -1) * circle,
                (np.expand_dims(radius, -1) + 0.2) * circle,
            ],
            axis=0,
        )
        X += self.noise * self.rng.normal(size=(n_data, 2))
        y_class = np.zeros(n_data, dtype=int)
        y_class[: n_data // 2] = 1
        # Convert to one-hot encoding
        y_onehot = np.eye(self.n_class)[y_class]
        return X, y_onehot


class DataGeneratorPeaks(DataGenerator):
    """Generator for the peaks dataset

    Data points in the domain [-3,+3] x [-3,+3] are classified using the peaks function

        f_peak(x,y) = 3 (1-x)^2 exp[-(x^2 + (y+1)^2)]
                    - 10 (1/5 x - x^3 - y^5) exp[-(x^2 + y^2)]
                    - 1/3 exp[-((x + 1)^2 + Y^2)]

    a point (x,y) is labelled to be of class k if the f_peak(x,y) lies in the range
    L_k ... L_{k+1}, where L = [-infinity,L_1,L_2,...,L_k,L_infinity] is a given set of levels.

    :arg n_train: total number of training samples to generate; must be a multiple of batchsize
    :arg n_test: total number of test samples to generate
    :arg batchsize: batch size
    """

    def __init__(self, n_train, n_test, batchsize):
        self.levels = [-4, -1, 2, 5, 10]
        super().__init__(2, len(self.levels) + 1, n_train, n_test, batchsize)

    def f_peak(self, X):
        """Peak function f_peak(x,y)

        :arg X: (x,y)-coordinates of points
        """
        return (
            3 * (1 - X[0]) ** 2 * np.exp(-(X[0] ** 2 + (X[1] + 1) ** 2))
            - 10
            * (0.2 * X[0] - X[0] ** 3 - X[1] ** 5)
            * np.exp(-(X[0] ** 2 + X[1] ** 2))
            - 1 / 3 * np.exp(-((X[0] + 1) ** 2 + X[1] ** 2))
        )

    def _generate_data(self, n_data):
        """Generate data points

        draw datapoints from peaks dataset

        :arg n_data: number of data points to generate

        Return tuple (X_data,y_data) where X_data is an array of shape (n_data,2) and
        y_data is a one-hot array of shape (n_data,n_c) with n_c being the number of classes
        """
        X = self.rng.uniform(low=-3, high=+3, size=(n_data, 2))
        values = self.f_peak(X.T)
        y_class = np.empty(n_data, dtype=int)
        all_levels = [-np.infty] + self.levels + [+np.infty]
        for j, (low, high) in enumerate(zip(all_levels[:-1], all_levels[1:])):
            y_class[np.where((low < values) & (values < high))] = j

        y_onehot = np.eye(self.n_class)[y_class]
        return X, y_onehot
