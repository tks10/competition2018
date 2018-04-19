from PIL import Image
import numpy as np
import glob


class DatasetLoader(object):
    def __init__(self, directory_paths, labels, init_size=(64, 64)):
        self._data = DatasetLoader.import_all_images(directory_paths, labels, init_size)

    @staticmethod
    def import_all_images(directory_paths, labels, init_size=(64, 64)):
        assert len(directory_paths) == len(labels), "directory_paths and labels must be same length."
        _data = DatasetLoader.import_images(directory_paths[0], labels[0], init_size)
        # Load additional datas if directory_paths have more than 2 items.
        if len(directory_paths) > 1:
            for directory_path, label in zip(directory_paths[1:], labels[1:]):
                _data += DatasetLoader.import_images(directory_path, label, init_size)
        return _data

    @staticmethod
    def import_images(directory_path, label, init_size=None):
        file_paths = glob.glob(directory_path + "/*")
        _images = []
        height, width = -1, -1
        # Load images from directory_path using generator.
        for image in DatasetLoader.image_generator(file_paths, init_size):
            assert (height == image.shape[0] and width == image.shape[1]) or height == -1,\
                "cannot import images which have a different resolution."
            height, width = image.shape[0], image.shape[1]
            _images.append(image)
        _labels = [label for _ in range(len(_images))]

        return DataSet(_images, _labels)

    @staticmethod
    def image_generator(file_paths, init_size=None):
        """

        `A generator which yields images deleted an alpha channel and resized.
         アルファチャネル削除、リサイズ(任意)処理を行った画像を返します

        Args:
            file_paths (list[string]): File paths you want load.
            init_size (tuple(int, int)): If having a value, images are resized by init_size.

        Yields:
            image (ndarray[width][height][channel]): processed image.

        """
        for file_path in file_paths:
            image = Image.open(file_path)
            if init_size is not None and init_size != image.size:
                image = image.resize(init_size)
            if image.mode == "RGBA":
                image = image.convert("RGB")
                # TODO(tks10): Deal with an alpha channel.
                # If original pixel's value aren't 255, contrary to expectations, the pixels may be not white.
            image = np.asarray(image)

            yield image

    def load_train_test(self, train_rate=0.8, shuffle=True, transpose_by_color=False):
        """

        `Load datasets splited into training set and test set.
         訓練とテストに分けられたデータセットをロードします．

        Args:
            train_rate (float): Training rate.
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]

        Returns:
            Training Set (Dataset), Test Set (Dataset)

        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        raw_data = self._data if not transpose_by_color else self._data.transpose_by_color()
        if shuffle:
            raw_data = raw_data.shuffle()
        train_size = int(len(self._data.images) * train_rate)
        data_size = int(len(self._data.images))
        _train_set = raw_data.perm(0, train_size)
        _test_set = raw_data.perm(train_size, data_size)

        return _train_set, _test_set

    def load_train_test_batch(self, train_rate=0.8, batch_size=20, shuffle=True, transpose_by_color=False):
        """

        `Load batches splited into training set and test set.
         訓練とテストに分けられたバッチデータをロードします．

        Args:
            train_rate (float): Training rate.
            batch_size (int): Batch size.
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]

        Returns:
            Training batches (ndarray), Test batches (ndarray)

        """
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("train_rate must be from 0.0 to 1.0.")
        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        train_set, test_set = self.load_train_test(train_rate, shuffle, transpose_by_color)
        train_res, test_res = [], []
        for start in range(0, train_set.length, batch_size):
            permed = train_set.perm(start, start+batch_size)
            train_res.append({"data": permed.images, "label": permed.labels})
        for start in range(0, test_set.length, batch_size):
            permed = train_set.perm(start, start+batch_size)
            test_res.append({"data": permed.images, "label": permed.labels})

        return np.asarray(train_res), np.asarray(test_res)

    def load_raw_dataset(self, shuffle=True, transpose_by_color=False):
        """

        `Load a raw dataset which has all datas.
         未加工のデータすべてをロードします.

        Args:
            shuffle (bool): If true, shuffle dataset.
            transpose_by_color (bool): If True, transpose images for chainer. [channel][width][height]

        Returns:
            Raw dataset (Dataset)

        """
        _res = self._data if not transpose_by_color else self._data.transpose_by_color()
        return _res if not shuffle else _res.shuffle()


class DataSet(object):
    TARGET = 0
    OTHERS = 1

    def __init__(self, images, labels):
        self._images = np.asarray(images)
        self._labels = np.asarray(labels)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def length(self):
        return len(self._images)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images))

    def __iadd__(self, other):
        _images = np.concatenate([self.images, other.images])
        _labels = np.concatenate([self.labels, other.labels])
        return DataSet(_images, _labels)

    def shuffle(self):
        _list = list(zip(self._images, self._labels))
        np.random.shuffle(_list)
        _images, _labels = zip(*_list)
        return DataSet(np.asarray(_images), np.asarray(_labels))

    def transpose_by_color(self):
        _image = self._images.transpose(0, 3, 1, 2)
        return DataSet(_image, self._labels)

    def perm(self, start, end):
        if end > len(self._images):
            end = len(self._images)
        return DataSet(self._images[start:end], self._labels[start:end])


if __name__ == "__main__":
    dataset_loader = DatasetLoader(["Images", "Images"], [DataSet.TARGET, DataSet.OTHERS])
    train, test = dataset_loader.load_train_test()
    train.print_information()
    test.print_information()


