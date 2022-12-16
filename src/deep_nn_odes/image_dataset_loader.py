"""Classes for reading labelled image datasets

Can be used to load the MNIST and CIFAR10 datasets from disk. 

The classes are derived from a common base class, which provides the following three key methods for
training neural networks for image classification:

* get_shuffled_batched_train_data() returns the batched images and labels as a list of minibatches;
    each image-minibatch is of the shape (batch_size,n_x,n_y,C) and each label-minibatch is of the
    shape (batch_size,). The data can be randomly shuffled, so that the network sees that data in
    a different order in each epoch.
* get_validation_data() returns the batched images and labels used for validation; the image batches
    are of shape (n_valid, n_x,n_y,C) and the labels of shape (n_valid,)
* get_test_data() returns the batched images and labels used for testing; the image batches
    are of shape (n_test, n_x,n_y,C) and the labels of shape (n_test,)

By default, the image width and height are rescaled to n_x = n_y = 32. These are the dimensions
of the CIFAR10 data, but note that the MNIST dataset consists of images of size 28 x 28.
"""

import os
import pickle
import numpy as np
import cv2


class ImageDatasetLoader:
    """Data loader base class.

    All dataset-specific classes are derived from this base class, which provides the
    fundamental functions for accessing the batched training, validation and test-data:
    """

    def __init__(
        self,
        n_x=None,
        n_y=None,
        n_channels=1,
        n_categories=None,
        n_train_valid=None,
        n_test=None,
        validation_split=0.2,
        normalise_images=True,
    ):
        """Initialise new instance

        :arg n_x: image width W
        :arg n_y: image height H
        :arg n_channels: number of channels C
        :arg n_categories: number of categories
        :arg n_train_valid: number of training + validation images
        :arg n_test: number of test images
        :arg validation_split: fraction of training images used for validation
        :arg normalise_images: normalise images by subtracting mean and dividing
            by standard deviation of training images?
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_channels = n_channels
        self.n_train_valid = n_train_valid
        self.n_test = n_test
        self.n_categories = n_categories
        self.validation_split = validation_split
        self.normalise_images = normalise_images
        # random number generator
        self.rng = np.random.default_rng(seed=2149187)
        self.train_valid_images = np.empty((1, 1, 1, 1))
        self.train_valid_labels = np.empty((1,))
        self.test_images = np.empty((1, 1, 1, 1))
        self.test_labels = np.empty((1,))

    def normalise_and_transpose(self):
        """Normalise images using the mean and standard deviation of the
        training dataset (including both training- and validation images) and transpose them to
        HWC, if necessary (recall that by default images are stored as CHW)."""
        if self.normalise_images:
            # subtract mean and divide by standard deviation of
            # training images
            avg = np.mean(self.train_valid_images, axis=(0, 2, 3)).reshape(
                (1, self.n_channels, 1, 1)
            )
            std = np.std(self.train_valid_images, axis=(0, 2, 3)).reshape(
                (1, self.n_channels, 1, 1)
            )
            self.train_valid_images = (self.train_valid_images - avg) / std
            self.test_images = (self.test_images - avg) / std
        # transpose to HWC format
        self.train_valid_images = self.train_valid_images.transpose([0, 2, 3, 1])
        self.test_images = self.test_images.transpose([0, 2, 3, 1])

    def get_shuffled_batched_train_data(self, batch_size, random_shuffle=True):
        """Return batched data for training.

        Returns two lists, containing the batched images and the batched labels.

        Each element of the lists consists of a minibatch of size batch_size.
        If n_train is not a multiply of batch_size, any remaining elements will be dropped.

        :arg batch_size: size of minibatches
        :arg random_shuffle: randomly shuffle the training dataset before batching?
        """
        n_train = int((1 - self.validation_split) * self.n_train_valid)
        idx_array = np.arange(0, n_train)
        if random_shuffle:
            self.rng.shuffle(idx_array)
        n_batches = n_train // batch_size
        assert n_batches > 0
        batched_images = np.split(
            self.train_valid_images[idx_array[: n_batches * batch_size]], n_batches
        )
        batched_labels = np.split(
            self.train_valid_labels[idx_array[: n_batches * batch_size]],
            n_batches,
        )  # pylint: disable=unsubscriptable-object
        return batched_images, batched_labels

    def get_validation_data(self):
        """Return validation data

        This includes all images that are not used for training. Returns two arrays,
        containing the batched images and the batched labels."""
        n_train = int((1 - self.validation_split) * self.n_train_valid)
        return self.train_valid_images[n_train:], self.train_valid_labels[n_train:]

    def get_test_data(self):
        """Return test data

        This includes all images that are not used for training. Returns two arrays,
        containing the batched images and the batched labels."""
        return self.test_images, self.test_labels


class MNISTDatasetLoader(ImageDatasetLoader):
    """Data loader for MNIST dataset.

    See https://deepai.org/dataset/mnist and http://yann.lecun.com/exdb/mnist/.
    The files can be obtained with wget https://data.deepai.org/mnist.zip.
    Download the files into the data directory and unzip them with
    gunzip FILENAME.gz.

    After unpacking the data directory should contain the following files:

        * train-images-idx3-ubyte
        * train-labels-idx1-ubyte
        * t10k-images-idx3-ubyte
        * t10k-labels-idx1-ubyte
    """

    def __init__(
        self,
        datadir="../data/mnist/",
        validation_split=0.2,
        resize=True,
        normalise_images=True,
    ):
        """Initialise new instance

        :arg datadir: directory containing the datset files
        :arg validation_split: fraction of training images used for validation
        :arg resize: resize images to 32x32?
        :arg normalise_images: normalise images by subtracting mean and dividing
            by standard deviation of training images?
        """
        super().__init__(
            n_x=32 if resize else 28,
            n_y=32 if resize else 28,
            n_channels=1,
            n_categories=10,
            n_train_valid=60000,
            n_test=10000,
            validation_split=validation_split,
            normalise_images=normalise_images,
        )
        self.resize = resize
        self.datadir = datadir
        # Read training images and labels
        self.train_valid_images = self._read_image_file(
            os.path.join(self.datadir, "train-images-idx3-ubyte")
        )
        self.train_valid_labels = self._read_label_file(
            os.path.join(self.datadir, "train-labels-idx1-ubyte")
        )
        # Check that the number of read images and labels is as expected
        assert self.n_train_valid == self.train_valid_images.shape[0]
        assert self.n_train_valid == self.train_valid_labels.shape[0]
        # Read test images and labels
        self.test_images = self._read_image_file(
            os.path.join(self.datadir, "t10k-images-idx3-ubyte")
        )
        self.test_labels = self._read_label_file(
            os.path.join(self.datadir, "t10k-labels-idx1-ubyte")
        )
        # Check that the number of read images and labels is as expected
        assert self.n_test == self.test_images.shape[0]
        assert self.n_test == self.test_labels.shape[0]
        # set categories
        self.categories = [str(j) for j in range(10)]
        self.normalise_and_transpose()

    def _read_image_file(self, filename):
        """Read images from a single file, using the format described in
        http://yann.lecun.com/exdb/mnist/ and https://deepai.org/dataset/mnist

        Returns an array of shape (n_images,1,n_x,n_y) and type float32

        :arg filename: name of file to read
        """
        with open(filename, "rb") as f:
            # read header
            header = f.read(16)
            magic = int.from_bytes(header[0:4], byteorder="big")
            assert magic == 2051
            n_images = int.from_bytes(header[4:8], byteorder="big")
            n_x = int.from_bytes(header[8:12], byteorder="big")
            n_y = int.from_bytes(header[12:16], byteorder="big")
            assert n_x == 28
            assert n_y == 28
            data = f.read(n_x * n_y * n_images)
            orig_images = np.reshape(
                np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 256,
                (n_images, n_x, n_y),
            )
            if self.resize:
                images = np.empty((n_images, self.n_x, self.n_x))
                for j in range(n_images):
                    images[j, :, :] = cv2.resize(  # pylint: disable=no-member
                        orig_images[j, :, :], (self.n_x, self.n_y)
                    )
            else:
                images = orig_images
            images = np.reshape(images, [n_images, 1, self.n_x, self.n_y])

            return images

    def _read_label_file(self, filename):
        """Read labels from a single file, using the format described in
        http://yann.lecun.com/exdb/mnist/

        Returns an array of shape (n_images,) and type uint32

        :arg filename: name of file to read
        """
        with open(filename, "rb") as f:
            # read header
            header = f.read(8)
            magic = int.from_bytes(header[0:4], byteorder="big")
            assert magic == 2049
            n_images = int.from_bytes(header[4:8], byteorder="big")
            data = f.read(n_images)
            labels = np.reshape(
                np.frombuffer(data, dtype=np.uint8),
                [
                    n_images,
                ],
            )
            return labels


class CIFAR10DatasetLoader(ImageDatasetLoader):
    """Loader for the CIFAR10 dataset.

    The dataset is described at https://www.cs.toronto.edu/~kriz/cifar.html.
    Download the CIFAR-10 python version and unpack it into the correct directory.

    After unpacking the data directory should contain the following files:

        * batches.meta
        * data_batch_1
        * data_batch_2
        * data_batch_3
        * data_batch_4
        * data_batch_5
        * test_batch
    """

    def __init__(
        self,
        datadir="../data/cifar-10-batches-py/",
        validation_split=0.2,
        normalise_images=True,
    ):
        """Initialise new instance

        :arg datadir: directory containing the datset files
        :arg validation_split: fraction of training images used for validation
        :arg normalise_images: normalise images by subtracting mean and dividing
            by standard deviation of training images?
        """
        super().__init__(
            n_x=32,
            n_y=32,
            n_channels=3,
            n_categories=10,
            n_train_valid=50000,
            n_test=10000,
            validation_split=validation_split,
            normalise_images=normalise_images,
        )
        self.datadir = datadir
        # Load training images
        self.train_valid_images = np.empty([50000, 3, 32, 32], dtype=np.float32)
        self.train_valid_labels = np.empty([50000], dtype=np.uint8)
        for idx in range(5):
            batch_images, batch_labels = self._unpickle(
                os.path.join(self.datadir, f"data_batch_{(idx+1):d}")
            )
            self.train_valid_images[
                10000 * idx : 10000 * (idx + 1), :, :, :
            ] = batch_images
            self.train_valid_labels[10000 * idx : 10000 * (idx + 1)] = batch_labels
        # Load test images
        self.test_images, self.test_labels = self._unpickle(
            os.path.join(self.datadir, "test_batch")
        )
        # Load descriptions
        filename = os.path.join(self.datadir, "batches.meta")
        with open(filename, "rb") as f:
            meta_data = pickle.load(f, encoding="bytes")
        self.categories = [
            str(category, encoding="utf8") for category in meta_data[b"label_names"]
        ]
        assert len(self.categories) == self.n_categories
        # Normalise and transpose images, if necessary
        self.normalise_and_transpose()

    def _unpickle(self, filename):
        """Unpickle CIFAR10 datafile, as described at https://www.cs.toronto.edu/~kriz/cifar.html

        :arg filename: name of file to load
        """
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        images = data[b"data"].reshape(10000, 3, 32, 32).astype(np.float32) / 256
        labels = np.asarray(data[b"labels"], dtype=np.uint8)
        return images, labels
