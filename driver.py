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
from tfutils import TFLoaderKaggleThread, TFLoaderDiaretdb1Thread, total_length


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Driver(object):
    def __init__(self, num_epochs, model_load_path, architecture, attention_names):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_epochs: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        """

        self.global_step = 0
        self.num_epochs = num_epochs
        self.mode_load_path = model_load_path

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)

        print('Put placeholder and define the reader thread.')
        self.height = c.MULTI_SCALES[c.SCALE_INDEX]
        self.width = c.MULTI_SCALES[c.SCALE_INDEX]
        self.batch_size = c.BATCH_SIZE

        if c.IS_TRAINING:
            batch_size = c.BATCH_SIZE
        else:
            batch_size = c.BATCH_SIZE * c.MULTI_TEST

        with tf.name_scope('data'):
            # Classification, kaggle
            self.data_input = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.height, self.width, 3]
            )
            self.label_input = tf.placeholder(
                dtype=tf.float32,
                shape=[None]
            )
            # Attention, diaretdb1
            self.attention_input = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.height, self.width, 3]
            )
            self.attention_map = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.height, self.width, 1]
            )

        # kaggle
        self.kaggle_queue = tf.FIFOQueue(
            capacity=20,
            dtypes=[tf.float32, tf.float32],
            shapes=[[self.height, self.width, 3], []]
        )
        self.kaggle_enqueue_op = self.kaggle_queue.enqueue_many([self.data_input, self.label_input])

        # diaretdb1
        self.diaretdb1_queue = tf.FIFOQueue(
            capacity=20,
            dtypes=[tf.float32, tf.float32],
            shapes=[[self.height, self.width, 3], [self.height, self.width, 1]]
        )
        self.diaretdb1_enqueue_op = self.diaretdb1_queue.enqueue_many([self.attention_input, self.attention_map])

        batch_data_tensor, batch_label_tensor = self.kaggle_queue.dequeue_many(batch_size)
        batch_att_tensor, batch_map_tensor = self.diaretdb1_queue.dequeue_many(c.ATTENTION_BATCH_SIZE)

        print(batch_data_tensor.shape)
        print(batch_label_tensor.shape)
        print(batch_att_tensor.shape)
        print(batch_map_tensor.shape)

        print('Init models...')
        self.model = ClassifierModel(self.sess,
                                     data_labels_tensors=(batch_data_tensor, batch_label_tensor,
                                                          batch_att_tensor, batch_map_tensor),
                                     multi_scales=c.MULTI_SCALES,
                                     architecture=architecture,
                                     attention_names=attention_names)

        print('Define graphs...')
        self.model.define_graph(c.IS_TRAINING)

        print('Init variables...')
        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess.run(tf.global_variables_initializer())

        print('Load pretrain weights...')
        if c.USE_PRETRAIN:
            self.model.use_pretrain(c.USE_PRETRAIN)

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
        c.DECAY_STEP = int(np.ceil(total_length / c.BATCH_SIZE))

        print('Training the network on gpu {} ...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        print('\tLearning rate = {}'.format(c.LEARNING_RATE))
        print('\tMulti scales = {}'.format(c.MULTI_SCALES))
        print('\tTraining image directory => {}'.format(c.TRAIN_FOLDER))
        print('\tValidate image directory => {}'.format(c.VALIDATE_IMAGES_FOLDERS))
        print('\tTesting image directory => {}'.format(c.TEST_FOLDER))
        print('\tBatch size = {}'.format(c.BATCH_SIZE))
        print('\tNumber of threads = {}'.format(c.NUM_THREADS))
        print('\tSaving model directory = {}'.format(c.MODEL_SAVE_DIR))
        print('\tSaving summary directory = {}'.format(c.SUMMARY_SAVE_DIR))
        print('\tDebug = {}'.format(c.DEBUG))
        print('\tSAVE FREQ = {}'.format(c.MODEL_SAVE_FREQ))
        print('\tAttention = {}'.format(c.ATTENTIONS))

        coord = tf.train.Coordinator()
        try:
            # Create 10 threads that run 'load_and_enqueue()'
            threads1 = [TFLoaderKaggleThread('Kaggle'+str(i), sess=self.sess, enqueue_op=self.kaggle_enqueue_op,
                                             coord=coord, data_input=self.data_input, label_input=self.label_input)
                        for i in range(c.NUM_THREADS)]
            threads2 = [TFLoaderDiaretdb1Thread('Diaretdb'+str(i), sess=self.sess, enqueue_op=self.diaretdb1_enqueue_op,
                                                coord=coord, data_input=self.attention_input,
                                                label_input=self.attention_map) for i in range(c.NUM_THREADS)]
            threads = threads1 + threads2
            # Start the threads and wait for all of them to stop.
            for t in threads:
                t.start()

            for epoch in range(self.num_epochs):
                self.global_step = self.model.train_step()
                if self.global_step % c.MODEL_SAVE_FREQ == 0:
                    print('-' * 30)
                    print('Saving models...')
                    self.saver.save(self.sess,
                                    os.path.join(c.MODEL_SAVE_DIR, 'model.ckpt'),
                                    global_step=self.global_step)
                    print('Saved models!')
                    print('-' * 30)

            # Close all threads
            for thread in threads:
                thread.is_run = False
            # Close queue
            self.sess.run(self.kaggle_queue.close(cancel_pending_enqueues=True))
            coord.join(threads, stop_grace_period_secs=10)

        except RuntimeError as e:
            coord.request_stop(e)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()

        self.sess.close()

    def inference(self, batch_data):
        self.sess.run(self.kaggle_enqueue_op, feed_dict={
            self.data_input: batch_data
        })
        logits = self.model.inference()
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
        print('\tNumber of threads = {}'.format(c.NUM_THREADS))
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

            _labels = np.zeros((batch_size * ensemble), dtype=np.float32)

            # inference on batch data (N*x, height, width, 3)
            # shapes (N * x, 1)
            self.sess.run(self.kaggle_enqueue_op, feed_dict={
                self.data_input: data,
                self.label_input: _labels
            })
            logits = self.model.inference()

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

        return logits_ensemble, true_labels

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
    parser.add_argument('--attention', type=str, nargs='*', choices=c.ATTENTIONS,
                        help='the name of tensor to add attention module, valid names are {}'.format(c.MULTI_SCALES))
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
    parser.add_argument('-n', '--num_threads', type=int, default=c.NUM_THREADS,
                        help='the number of crops when testing.')
    parser.add_argument('-l', '--load_path', type=str, default='',
                        help='the number of loading models.')
    parser.add_argument('--architecture', type=str, default='resnet_v2_50',
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
    parser.add_argument('-p', '--pretrain', type=str, default='',
                        help='the flag to control whether to use ImageNet pretrain model or not.')
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
    attentions = None
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
        if opt is 'num_threads':
            c.NUM_THREADS = int(arg)
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
        if opt is 'attention':
            attentions = arg
            c.ATTENTIONS = attentions
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
        scale = 'Scale_{}'.format('_'.join([str(x) for x in c.MULTI_SCALES]))
        attention = '-'.join(attentions)
        c.set_save_dir('{}-{}'.format(scale, attention))
    driver = Driver(num_epochs, load_path, architecture, attentions)
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
