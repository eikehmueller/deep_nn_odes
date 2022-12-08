"""Classes for reading labelled image data"""

import os
import pickle
import numpy as np


class DataLoader:
    """Data loader base class"""

    def __init__(
        self,
        n_x=None,
        n_y=None,
        n_channels=1,
        n_categories=None,
        n_train=None,
        n_test=None,
    ):
        """Initialise new instance

        :arg n_x: image size in horizontal direction
        :arg n_y: image size in vertical direction
        :arg n_channels: number of channels
        :arg n_categories: number of categories
        :arg n_train: number of training images
        :arg n_test: number of test images
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_channels = n_channels
        self.n_train = n_train
        self.n_test = n_test
        self.n_categories = n_categories


class MNISTDataLoader(DataLoader):
    def __init__(self, datadir="../data/mnist/"):
        super().__init__(
            n_x=28, n_y=28, n_channels=1, n_categories=10, n_train=60000, n_test=10000
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

    def _read_image_file(self, filename):
        """Read images from a single file, using the format described in
        http://yann.lecun.com/exdb/mnist/

        Returns an array of shape (n_images,n_x,n_y,1) and type float32

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
                np.frombuffer(data, dtype=np.uint8) / 256, [n_images, n_x, n_y, 1]
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


class CIFAR10DataLoader(DataLoader):
    def __init__(self, datadir="../data/cifar-10-batches-py/"):
        super().__init__(
            n_x=32, n_y=32, n_channels=3, n_categories=10, n_train=50000, n_test=10000
        )
        self.datadir = datadir
        # Load training images
        self.train_images = np.empty([50000, 32, 32, 3])
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

    def _unpickle(self, filename):
        """Unpickle CIFAR10 datafile, as described at https://www.cs.toronto.edu/~kriz/cifar.html

        :arg filename: name of file to load
        """
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        images = np.transpose(
            data[b"data"].reshape(10000, 3, 32, 32) / 256, [0, 2, 3, 1]
        ).astype(np.float32)
        labels = np.asarray(data[b"labels"], dtype=np.uint8)
        return images, labels
