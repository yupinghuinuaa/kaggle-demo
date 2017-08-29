import cv2
import os
import numpy as np
from process_data import check_images_match_labels, load_label
import constants as c


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class DataIndexLoader(object):
    """
    Singleton class, which loads the image paths and corresponding labels of each mini-batch
    and does not load the real data.
    """
    def __init__(self):
        self.classes_map_images = self.classes2images()
        self.total_images_labels = self.images_labels_list()
        self.total_length = len(self.total_images_labels)
        self.total_index = 0
        self.num_classes = len(self.classes_map_images.keys())
        self.classes_size = np.zeros(self.num_classes, dtype=np.int32)

    def __call__(self, batch_size, t):
        """
        :param batch_size:
        :param t:
        :return:
        """
        # determine the scale
        c.SCALE_INDEX = np.random.randint(0, len(c.MULTI_SCALES))

        # sample the (image path, label)
        images_labels = []

        start = self.total_index
        end = self.total_index + batch_size
        for i in range(start, end):
            if i == self.total_length:
                np.random.shuffle(self.total_images_labels)
            idx = i % self.total_length
            image_name = self.total_images_labels[idx][0]
            label = self.total_images_labels[idx][1]
            image_path = os.path.join(c.TRAIN_IMAGES_FOLDERS[c.SCALE_INDEX], image_name)
            images_labels.append((image_path, label))

        self.total_index = end % self.total_length

        # for image, label in images_labels:
        #     print('{}, {}'.format(image, label))
        # print('{} / {}'.format(self.total_index, self.total_length))

        if c.SHUFFLE:
            np.random.shuffle(images_labels)
        return images_labels

    @staticmethod
    def average_sample_rate(t):
        """
        t = [0: 10000],     1.0 - 0.9
        t = [10000: 20000], 0.9 - 0.8
        t = [20000: 30000], 0.8 - 0.7
        t = [30000: 40000], 0.7 - 0.6
        :param t: Iteration step
        :return: The probability to sample on average
        """
        return np.exp(-0.00001 * t)

    @staticmethod
    def classes2images():
        """
        return the following data information:
        labels_maps_iamges {
            int(0): {
                images: [xxx.jpeg, xxx.jpeg....]
                index: 0
                length: the length of images
            }
            int(1): {
                ....
            }
            ....
        }
        :return: the labels map to images.
        :type: dict
        """
        labels_map_images = dict()

        images_labels_list = load_label(c.TRAIN_LABELS)
        for (image, label) in images_labels_list:
            label = int(label)
            if label not in labels_map_images:
                labels_map_images[label] = dict()
                labels_map_images[label]['images'] = []
                labels_map_images[label]['index'] = 0
                labels_map_images[label]['length'] = 0

            labels_map_images[label]['images'].append(image)
            labels_map_images[label]['length'] += 1

        return labels_map_images

    @staticmethod
    def images_labels_list():
        images_labels = load_label(c.TRAIN_LABELS)
        for i in range(len(images_labels)):
            images_labels[i][1] = int(images_labels[i][1])
        np.random.shuffle(images_labels)
        return images_labels


def _flip_rotate(image):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
    rotate_time = np.random.randint(0, 4)
    image = np.rot90(image, rotate_time)
    return image


def pre_processing(image_path, flip_rotate):
    image = cv2.imread(image_path)
    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    if not image.shape == (crop_size, crop_size, 3):
        image = cv2.resize(image, (crop_size, crop_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(dtype=np.float32)

    if flip_rotate:
        image = _flip_rotate(image)

    return image


def load_batch(batch_size=c.BATCH_SIZE, step=0, flip_rotate=True):
    data_loader = DataIndexLoader()
    images_labels = data_loader(batch_size=batch_size, t=step)

    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    batch_data = np.empty([batch_size, crop_size, crop_size, 3], dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)

    for idx, (image, label) in enumerate(images_labels):
        batch_data[idx] = pre_processing(image, flip_rotate)
        batch_labels[idx] = label

    return batch_data, batch_labels


def load_images_labels(image_dir, label_path):
    assert os.path.isdir(image_dir), 'There is no image directory {}!'.format(image_dir)
    assert os.path.exists(label_path), 'There is no label file {}!'.format(label_path)

    images_paths = sorted(os.listdir(image_dir))
    images_labels = load_label(label_path)

    assert check_images_match_labels(images_labels, images_paths), 'images_labels dose not match to image_paths!'

    for i in range(len(images_labels)):
        image, label = images_labels[i]
        image_path = os.path.join(image_dir, image)
        images_labels[i] = (image_path, int(label))

    return images_labels


def load_batch_by_images_labels(images_labels):
    batch_size = len(images_labels)
    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    batch_data = np.zeros([batch_size, crop_size, crop_size, 3], dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)

    for idx, (image, label) in enumerate(images_labels):
        batch_data[idx] = pre_processing(image, True)
        batch_labels[idx] = label

    return batch_data, batch_labels


def fetch_test_batch(batch_images_labels, flip_rotate=False):
    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    batch_size = len(batch_images_labels)
    batch_data = np.empty([batch_size * c.MULTI_TEST, crop_size, crop_size, 3], dtype=np.float32)
    batch_labels = np.zeros((batch_size,))

    for i, (image, label) in enumerate(batch_images_labels):
        batch_labels[i] = label

        # original image (no fli p and no rotation)
        original_image = pre_processing(image, flip_rotate=False)
        #
        # args to control whether to flip and rotate
        flip_args = [False, True]
        rotate_args = [0, 1, 2, 3]
        #
        j = 0
        for _flip in flip_args:
            for _rotate in rotate_args:
                process_image = np.fliplr(original_image) if _flip else original_image.copy()
                process_image = np.rot90(process_image, _rotate)
                batch_data[i*c.MULTI_TEST + j] = process_image
                j += 1

        batch_data[i*c.MULTI_TEST] = pre_processing(image, flip_rotate=False)

        # left c.MULTI_TEST - 8 random to flip and rotate
        for j in range(8, c.MULTI_TEST):
            batch_data[i*c.MULTI_TEST + j] = pre_processing(image, True)

    return batch_data, batch_labels


if __name__ == '__main__':

    # for i in range(500):
    #     data, labels = load_batch(10)
    #     print('Iteration = {}, data = {}, labels = {}'.format(i, data.shape, labels.shape))

    # images_labels = load_images_labels(c.VALIDATE_IMAGES_FOLDER, c.VALIDATE_LABELS)
    #
    # for image, label in images_labels[0:100]:
    #     print(image, label)

    # for i in range(40000):
    #     data, label = load_batch(10, step=0)
    #     print(data.shape, label.shape)
    dataLoader = DataIndexLoader()
    for (image, label) in dataLoader.total_images_labels:
        print(image, label)
    # for (key, value) in dataLoader.classes_map_images.items():
    #     print(key, value['length'])
