import tensorflow as tf
import numpy as np
import argparse
import os
import sys

from models import ClassifierModel
from utils import load_batch, load_images_labels, fetch_test_batch
import constants as c
from metrics_utils import metrics
from evaluate import evaluate_kappa_accuracy

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Driver(object):
    def __init__(self, num_epochs, model_load_path, num_crops, architecture):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_epochs: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        @param num_crops: The number of recursive generations to produce when testing. Recursive
                             generations use previous generations as input to predict further into
                             the future.
        """

        self.global_step = 0
        self.num_epochs = num_epochs
        self.num_crops = num_crops
        self.mode_load_path = model_load_path

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        print('Init models...')
        self.model = ClassifierModel(self.sess, num_crops=c.NUM_CROPS,
                                     multi_scales=c.MULTI_SCALES,
                                     architecture=architecture)

        print('Define graphs...')
        self.model.define_graph(c.IS_TRAINING)

        print('Init variables...')
        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess.run(tf.global_variables_initializer())

        if c.USE_PRETRAIN:
            print('\tUsing ImageNet pre-trained {}'.format(architecture))
            self.model.init_imagenet()

        # for variable in tf.global_variables():
        #     print(self.sess.run(tf.is_variable_initialized(variable)), variable)

        # if load path specified, load a saved model
        if model_load_path:
            self.saver.restore(self.sess, model_load_path)
            print('Model restored from ' + model_load_path)

        print('Init successfully!')

    def train(self):
        """
        Runs a training loop on the model networks.
        """

        print('Training the network on gpu {} ...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        print('\tLearning rate = {}'.format(c.LEARNING_RATE))
        print('\tMulti scales = {}'.format(c.MULTI_SCALES))
        print('\tTraining image directory => {}'.format(c.TRAIN_FOLDER))
        print('\tValidate image directory => {}'.format(c.VALIDATE_IMAGES_FOLDERS))
        print('\tTesting image directory => {}'.format(c.TEST_FOLDER))
        print('\tBatch size = {}'.format(c.BATCH_SIZE))
        print('\tNumber of crops = {}'.format(self.num_crops))
        print('\tSaving model directory = {}'.format(c.MODEL_SAVE_DIR))
        print('\tSaving summary directory = {}'.format(c.SUMMARY_SAVE_DIR))
        print('\tDebug = {}'.format(c.DEBUG))

        for epoch in range(self.num_epochs):

            # update generator
            data, labels = load_batch(c.BATCH_SIZE, step=0, flip_rotate=True)

            self.global_step = self.model.train_step(data, labels)

            # save the models
            if self.global_step % c.MODEL_SAVE_FREQ == 0:
                print('-' * 30)
                print('Saving models...')
                self.saver.save(self.sess,
                                os.path.join(c.MODEL_SAVE_DIR, 'model.ckpt'),
                                global_step=self.global_step)
                print('Saved models!')
                print('-' * 30)

            # test generator model
            # if self.global_step % c.VALIDATE_FREQ == 0:
            #     self.test(c.VALIDATE_IMAGES_FOLDER, c.VALIDATE_LABELS, batch_size=1)

        self.sess.close()

    def inference(self, batch_data):
        logits = self.model.inference(batch_data)
        return logits

    def test(self, images_dir, label_path, batch_size=1):
        """
        Runs one test step on the generator network.
        """
        print('Testing the network on gpu {} ...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        print('\tCheckpoints = {}'.format(self.mode_load_path))
        print('\tMulti scales = {}'.format(c.MULTI_SCALES))
        print('\tTesting image directory => {}'.format(images_dir))
        print('\tLabel path => {}'.format(label_path))
        print('\tIs training = {}'.format(c.IS_TRAINING))
        print('\tBatch size = {}'.format(c.BATCH_SIZE))
        print('\tNumber of crops = {}'.format(self.num_crops))
        print('\tDebug = {}'.format(c.DEBUG))
        print('\tMulti test = {}'.format(c.MULTI_TEST))

        images_labels = load_images_labels(images_dir, label_path)
        length = len(images_labels)
        true_labels = np.empty(shape=(length,), dtype=np.int32)
        ensemble = c.MULTI_TEST
        logits_ensemble = np.empty(shape=(length, ensemble), dtype=np.float32)

        num_batches = int(np.ceil(length / batch_size))
        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, length)
            data, labels = fetch_test_batch(images_labels[start: end])

            # inference on batch data (N*x, height, width, 3)
            # shapes (N * x, 1)
            logits = self.model.inference(data)

            # reshape to (N, x)
            logits = np.reshape(logits, newshape=(-1, ensemble))

            for i in range(start, end):
                logits_ensemble[i] = logits[i-start]
                true_labels[i] = images_labels[i][1]
                print('Image = {}, Ensemble = {},, True label = {}'.format(i,  logits_ensemble[i], true_labels[i]))

        self.sess.close()

        result_save_path = os.path.join(c.RES_SAVE_DIR, os.path.split(self.mode_load_path)[-1])
        np.savez_compressed(result_save_path, pred_ensemble=None,
                            logits_ensemble=logits_ensemble,
                            true_labels=true_labels)

        return true_labels

    def evaluate(self, images_dir, label_path, metric_types, batch_size=1):
        # 1. compute the pred labels and load true labels.
        logits_ensemble, true_labels = self.test(images_dir, label_path, batch_size)

        # 2. evaluate
        kappa, accuracy, mean_accuracy, class_accuracy, pred_labels = evaluate_kappa_accuracy(
            logits_ensemble=logits_ensemble,
            true_labels=true_labels,
            preds_ensemble=None
        )
        return kappa, accuracy, mean_accuracy, class_accuracy, pred_labels

    def debug(self):
        data, labels = load_batch(c.BATCH_SIZE)
        self.model.debug(data, labels)


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')

    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the device id of gpu.')
    parser.add_argument('-c', '--crop', type=int, nargs='*', choices=c.MULTI_SCALES, required=True,
                        help='the crop size, valid crop sizes are {}, '
                             'represents multi scale training'.format(c.MULTI_SCALES))
    parser.add_argument('-m', '--metric', type=str, nargs='*', default=c.METRIC_ACCURACY, choices=metrics.keys(),
                        help='the metric type of evaluation must be {}'.format(metrics.keys()))
    parser.add_argument('-r', '--train_dir', type=str, default=c.TRAIN_FOLDER,
                        help='the path of training folder.')
    parser.add_argument('-v', '--val_dir', type=str, default=c.VALIDATE_FOLDER,
                        help='the path of validating folder.')
    parser.add_argument('-t', '--test_dir', type=str, default='',
                        help='the path of testing folder.')
    parser.add_argument('-e', '--epochs', type=int, default=50000,
                        help='the total number of epochs, default is 50000.')
    parser.add_argument('-n', '--num_crops', type=int, default=c.NUM_CROPS,
                        help='the number of crops when testing.')
    parser.add_argument('-l', '--load_path', type=str, default='',
                        help='the number of loading models.')
    parser.add_argument('-a', '--architecture', type=str, default='resnet_v2_50',
                        help='the architecture of network, '
                             'there are resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200')
    parser.add_argument('--stats_freq', type=int, default=c.STATS_FREQ,
                        help='the frequency of displaying loss.')
    parser.add_argument('--summary_freq', type=int, default=c.SUMMARY_FREQ,
                        help='the frequency of saving summaries.')
    parser.add_argument('--validate_freq', type=int, default=c.VALIDATE_FREQ,
                        help='the frequency of validating.')
    parser.add_argument('--model_save_freq', type=int, default=c.MODEL_SAVE_FREQ,
                        help='the frequency of model snapshot.')
    parser.add_argument('-d', '--debug', action='store_true', help='the flag to control whether to debug or not.')
    parser.add_argument('-s', '--save', type=str, default='', help='the save name.')
    parser.add_argument('-b', '--batch', type=int, default=10,
                        help='set the batch size, default is 10.')
    parser.add_argument('-p', '--pretrain', action='store_true', help='the flag to control whether to use ImageNet '
                                                                      'pretrain model or not.')
    parser.add_argument('--multi_test', type=int, default=1, help='whether to use multi test or not')
    return parser.parse_args()


def main():
    ##
    # Handle command line input.
    ##

    load_path = None
    test_only = None
    num_epochs = None
    architecture = None
    save_name = None
    num_crops = c.NUM_CROPS
    metric_types = [c.METRIC_ACCURACY]

    # Args with Namespace
    args = parser_args()
    # Convert Namespace to dict by built-in function vars()
    for opt, arg in vars(args).items():
        if opt is 'gpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = arg
        if opt is 'train_dir':
            c.set_train_dir(arg)
        if opt is 'val_dir':
            c.set_validate_dir(arg)
        if opt is 'test_dir' and arg:
            c.set_test_dir(arg)
            test_only = True
            c.IS_TRAINING = False
        if opt is 'architecture':
            architecture = arg
        if opt is 'epochs':
            num_epochs = arg
        if opt is 'num_crops':
            c.NUM_CROPS = arg
            num_crops = c.NUM_CROPS
        if opt is 'load_path':
            load_path = arg
        if opt is 'stats_freq':
            c.STATS_FREQ = arg
        if opt is 'summary_freq':
            c.SUMMARY_FREQ = arg
        if opt is 'validate_freq':
            c.VALIDATE_FREQ = arg
        if opt is 'model_save_freq':
            c.MODEL_SAVE_FREQ = arg
        if opt is 'debug':
            c.DEBUG = arg
        if opt is 'metric':
            metric_types = arg
        if opt is 'crop':
            c.MULTI_SCALES = arg
        if opt is 'batch':
            assert arg > 0, 'batch size {} must > 0.'.format(arg)
            c.BATCH_SIZE = int(arg)
        if opt is 'pretrain':
            c.USE_PRETRAIN = arg
        if opt is 'multi_test':
            c.MULTI_TEST = arg
        if opt is 'save' and arg:
            save_name = arg

    ##
    # Init and run the predictor
    ##
    if save_name:
        c.set_save_dir(save_name)
    else:
        c.set_save_dir('Scale_{}'.format('_'.join([str(x) for x in c.MULTI_SCALES])))
    driver = Driver(num_epochs, load_path, num_crops, architecture)
    if test_only:
        assert load_path, 'In only testing setting, load_path {} must be specific and can not be None'.format(load_path)
        print('test = {}'.format(c.TEST_IMAGES_FOLDERS[c.SCALE_INDEX]))
        driver.evaluate(c.TEST_IMAGES_FOLDERS[c.SCALE_INDEX], c.TEST_LABELS,
                        metric_types=metric_types, batch_size=c.BATCH_SIZE)
    else:
        if c.DEBUG:
            c.set_save_dir('Debug')
            driver.debug()
            sys.exit(0)
        driver.train()

if __name__ == '__main__':
    main()
