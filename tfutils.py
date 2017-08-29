import numpy as np
import threading
import os
import cv2

import constants as c
from utils import DataIndexLoader

height = c.MULTI_SCALES[c.SCALE_INDEX]
width = c.MULTI_SCALES[c.SCALE_INDEX]
batch_size = c.BATCH_SIZE
attention_batch_size = c.ATTENTION_BATCH_SIZE

dataIndexLoader = DataIndexLoader()

# kaggle training data
total_index = 0
total_length = dataIndexLoader.total_length
total_images_labels = dataIndexLoader.total_images_labels.copy()
total_images_labels.sort()

# diaretdb1_v_1_1 training data
diaretdb1_images = os.listdir(c.TRAIN_LESION_IMAGES)
diaretdb1_length = len(diaretdb1_images)
diaretdb1_index = 0

kaggle_lock = threading.Lock()
diaretdb1_lock = threading.Lock()


flip_rotates_args = [(False, 0), (False, 1), (False, 2), (False, 3),
                     (True, 0), (True, 1), (True, 2), (True, 3)]


def get_kaggle_images_labels():
    global total_index, total_length, total_images_labels
    # kaggle images and labels
    # sample the (image path, label)
    images_labels = []
    start = total_index
    end = total_index + batch_size
    for i in range(start, end):
        if i == total_length:
            np.random.shuffle(total_images_labels)

        idx = i % total_length
        image_name = total_images_labels[idx][0]
        label = total_images_labels[idx][1]
        image_path = os.path.join(c.TRAIN_IMAGES_FOLDERS[c.SCALE_INDEX], image_name)
        images_labels.append((image_path, label))

    total_index = end % total_length

    # for image, label in images_labels:
    #     print('{}, {}'.format(image, label))
    # print('{} / {}'.format(total_index, total_length))

    if c.SHUFFLE:
        np.random.shuffle(images_labels)
    return images_labels


def get_diaretdb1_image():
    global diaretdb1_images, diaretdb1_length, diaretdb1_index

    image_paths = []
    start = diaretdb1_index
    end = diaretdb1_index + attention_batch_size
    for i in range(start, end):
        if i == diaretdb1_length:
            np.random.shuffle(diaretdb1_images)

        idx = i % diaretdb1_length
        image_name = diaretdb1_images[idx]
        image_paths.append((os.path.join(c.TRAIN_LESION_IMAGES, image_name),
                            os.path.join(c.TRAIN_LESION_MAPS, image_name)))
        # print(image_name)
    diaretdb1_index = end % diaretdb1_length

    return image_paths


def pre_processing(image_path, is_color=True, flip=False, rotate=0):
    if is_color:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
    if flip:
        image = np.fliplr(image)
    if rotate > 0:
        image = np.rot90(image, rotate)
    image = image.astype(dtype=np.float32)
    return image


def load_batch(images_labels):
    global flip_rotates_args

    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    batch_data = np.zeros([batch_size, crop_size, crop_size, 3], dtype=np.float32)
    batch_labels = np.zeros((batch_size,), dtype=np.float32)

    # read kaggle data
    for idx, (image_path, label) in enumerate(images_labels):
        _flip, _rotate = flip_rotates_args[np.random.randint(0, 8)]
        image = pre_processing(image_path, is_color=True, flip=_flip, rotate=_rotate)
        batch_data[idx] = image[...]
        batch_labels[idx] = label

    return batch_data, batch_labels


def load_diaretdb1_batch(images_labels):
    global flip_rotates_args

    _batch_size = len(images_labels)
    crop_size = c.MULTI_SCALES[c.SCALE_INDEX]
    batch_data = np.zeros([_batch_size, crop_size, crop_size, 3], dtype=np.float32)
    batch_labels = np.zeros([_batch_size, crop_size, crop_size, 1], dtype=np.float32)

    for idx, (image_path, lesion_path) in enumerate(images_labels):
        _flip, _rotate = flip_rotates_args[np.random.randint(0, 8)]
        image = pre_processing(image_path, is_color=True, flip=_flip, rotate=_rotate)
        lesion = pre_processing(lesion_path, is_color=False, flip=_flip, rotate=_rotate)

        batch_data[idx] = image
        batch_labels[idx] = lesion

        # print(image_path, lesion_path)

    # print(batch_data.shape)
    # print(batch_labels.shape)
    return batch_data, batch_labels


class TFLoaderKaggleThread(threading.Thread):

    def __init__(self, thread_name, sess, enqueue_op, coord, data_input, label_input):
        self.sess = sess
        self.enqueue_op = enqueue_op
        self.coord = coord
        self.data_input = data_input
        self.label_input = label_input
        self.is_run = True
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        global kaggle_lock

        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                kaggle_lock.acquire()
                try:
                    images_labels = get_kaggle_images_labels()
                finally:
                    kaggle_lock.release()
                batch_data, batch_labels = load_batch(images_labels)
                self.sess.run(self.enqueue_op, feed_dict={
                    self.data_input: batch_data,
                    self.label_input: batch_labels,
                })

                if not self.is_run:
                    self.coord.request_stop()


class TFLoaderDiaretdb1Thread(threading.Thread):

    def __init__(self, thread_name, sess, enqueue_op, coord, data_input, label_input):
        self.sess = sess
        self.enqueue_op = enqueue_op
        self.coord = coord
        self.data_input = data_input
        self.label_input = label_input
        self.is_run = True
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        global diaretdb1_lock

        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                diaretdb1_lock.acquire()
                try:
                    images_labels = get_diaretdb1_image()
                finally:
                    diaretdb1_lock.release()
                batch_data, batch_labels = load_diaretdb1_batch(images_labels)
                self.sess.run(self.enqueue_op, feed_dict={
                    self.data_input: batch_data,
                    self.label_input: batch_labels,
                })

                if not self.is_run:
                    self.coord.request_stop()


if __name__ == '__main__':
    # Attention, diaretdb1
    import tensorflow as tf

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    height, width = 720, 720
    with tf.Session() as sess:
        attention_input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, height, width, 3]
        )
        attention_map = tf.placeholder(
            dtype=tf.float32,
            shape=[None, height, width, 1]
        )
        diaretdb1_queue = tf.FIFOQueue(
            capacity=20,
            dtypes=[tf.float32, tf.float32],
            shapes=[[height, width, 3], [height, width, 1]]
        )
        diaretdb1_enqueue_op = diaretdb1_queue.enqueue_many([attention_input, attention_map])
        input_data, map_data = diaretdb1_queue.dequeue_many(attention_batch_size)

        add_1 = input_data + 1

        coord = tf.train.Coordinator()
        threads = []
        for i in range(5):
            thread = TFLoaderDiaretdb1Thread('Diaretdb-' + str(i), sess=sess, coord=coord,
                                             enqueue_op=diaretdb1_enqueue_op,
                                             data_input=attention_input, label_input=attention_map)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for i in range(100):
            add_1_output = sess.run(add_1)
            print(i, add_1_output.max())

        coord.join(threads)
    images_labels = get_diaretdb1_image()
    batch_data, batch_labels = load_diaretdb1_batch(images_labels)
    print(batch_data.shape)
    print(batch_labels.shape)
