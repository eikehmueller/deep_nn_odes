"""Classes for reading labelled image data"""

import os
import pickle
import numpy as np


class ImageDatasetLoader:
    """Data loader base class"""

    def __init__(
        self,
        n_x=None,
        n_y=None,
        n_channels=1,
        n_categories=None,
        n_train=None,
        n_test=None,
        channels_first=True,
        normalise_images=True,
    ):
        """Initialise new instance

        :arg n_x: image width W
        :arg n_y: image height H
        :arg n_channels: number of channels C
        :arg n_categories: number of categories
        :arg n_train: number of training images
        :arg n_test: number of test images
        :arg channels_first: if this is true, then individual images will be stored as
               CHW, otherwise as HWC.
        :arg normalise_images: normalise images by subtracting mean and dividing by standard deviation
            of training images?
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_channels = n_channels
        self.n_train = n_train
        self.n_test = n_test
        self.n_categories = n_categories
        self.channels_first = channels_first
        self.normalise_images = normalise_images
        # random number generator
        self.rng = np.random.default_rng(seed=2149187)

    def normalise_and_transpose(self):
        """Normalise images using the mean and standard deviation of the
        training dataset and transpose them to HWC, if necessary (recall that by default images
        are stored as CHW)."""
        if self.normalise_images:
            # subtract mean and divide by standard deviation of
            # training images
            avg = np.mean(self.train_images, axis=(0, 2, 3)).reshape(
                (1, self.n_channels, 1, 1)
            )
            std = np.std(self.train_images, axis=(0, 2, 3)).reshape(
                (1, self.n_channels, 1, 1)
            )
            self.train_images = (self.train_images - avg) / std
            self.test_images = (self.test_images - avg) / std
        if not self.channels_first:
            # transpose to HWC, if necessary
            self.train_images = self.train_images.transpose([0, 2, 3, 1])
            self.test_images = self.test_images.transpose([0, 2, 3, 1])

    def get_shuffled_batched_train_data(self, batch_size, random_shuffle=True):
        """Return batched data for training.

        Returns two lists, containing the batched images and the batched labels.

        Each element of the lists consists of a minibatch of size batch_size.
        If n_train is not a multiply of batch_size, any remaining elements will be dropped.

        :arg batch_size: size of minibatches
        :arg random_shuffle: randomly shuffle the training dataset before batching?
        """
        idx_array = np.arange(0, self.n_train)
        if random_shuffle:
            self.rng.shuffle(idx_array)
        n_batches = self.n_train // batch_size
        assert n_batches > 0
        batched_images = np.split(
            self.train_images[idx_array[: n_batches * batch_size]], n_batches
        )
        batched_labels = np.split(
            self.train_labels[idx_array[: n_batches * batch_size]], n_batches
        )
        return batched_images, batched_labels


class MNISTDatasetLoader(ImageDatasetLoader):
    """Data loader for MNIST dataset.

    See https://deepai.org/dataset/mnist and http://yann.lecun.com/exdb/mnist/.
    The files can be obtained with wget https://data.deepai.org/mnist.zip.
    Downloaded the files into the data directory and unzip them with
    gunzip FILENAME.gz.
    """

    def __init__(
        self, datadir="../data/mnist/", channels_first=True, normalise_images=True
    ):
        """Initialise new instance

        :arg datadir: directory containing the datset files
        :arg channels_first: store the images is CHW format?
        :arg normalise_images: normalise images by subtracting mean and dividing by standard deviation
            of training images?
        """
        super().__init__(
            n_x=28,
            n_y=28,
            n_channels=1,
            n_categories=10,
            n_train=60000,
            n_test=10000,
            channels_first=channels_first,
            normalise_images=normalise_images,
        )
        self.datadir = datadir
        # Read training images and labels
        self.train_images = self._read_image_file(
            os.path.join(self.datadir, "train-images-idx3-ubyte")
        )
        self.train_labels = self._read_label_file(
            os.path.join(self.datadir, "train-labels-idx1-ubyte")
        )
        # Check that the number of read images and labels is as expected
        assert self.n_train == self.train_images.shape[0]
        assert self.n_train == self.train_labels.shape[0]
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
            assert n_x == self.n_x
            assert n_y == self.n_y
            data = f.read(n_x * n_y * n_images)
            images = np.reshape(
                np.frombuffer(data, dtype=np.uint8) / 256, [n_images, 1, n_x, n_y]
            ).astype(np.float32)
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
    """

    def __init__(
        self,
        datadir="../data/cifar-10-batches-py/",
        channels_first=True,
        normalise_images=True,
    ):
        """Initialise new instance

        :arg datadir: directory containing the datset files
        :arg channels_first: store the images is CHW format?
        :arg normalise_images: normalise images by subtracting mean and dividing by standard deviation
            of training images?
        """
        super().__init__(
            n_x=32,
            n_y=32,
            n_channels=3,
            n_categories=10,
            n_train=50000,
            n_test=10000,
            channels_first=channels_first,
            normalise_images=normalise_images,
        )
        self.datadir = datadir
        # Load training images
        self.train_images = np.empty([50000, 3, 32, 32])
        self.train_labels = np.empty([50000])
        for idx in range(5):
            batch_images, batch_labels = self._unpickle(
                os.path.join(self.datadir, f"data_batch_{(idx+1):d}")
            )
            self.train_images[10000 * idx : 10000 * (idx + 1), :, :, :] = batch_images
            self.train_labels[10000 * idx : 10000 * (idx + 1)] = batch_labels
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
