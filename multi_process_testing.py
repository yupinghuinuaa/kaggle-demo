import tensorflow as tf
import numpy as np
import argparse
import os
from multiprocessing import Queue, Process
import shelve
import time
import progressbar

from models import ClassifierModel
from utils import load_batch, load_images_labels, fetch_test_batch
import constants as c
from tfutils import TFLoaderKaggleThread, TFLoaderDiaretdb1Thread
from metrics_utils import metrics
from evaluate import evaluate_kappa_accuracy

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
        pass

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

        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        print('Testing the network on gpu {} ...'.format(gpu_id))
        print('\tCheckpoints = {}'.format(self.mode_load_path))
        print('\tMulti scales = {}'.format(c.MULTI_SCALES))
        print('\tTesting image directory => {}'.format(images_dir))
        print('\tLabel path => {}'.format(label_path))
        print('\tIs training = {}'.format(c.IS_TRAINING))
        print('\tBatch size = {}'.format(c.BATCH_SIZE))
        print('\tNumber of threads = {}'.format(c.NUM_THREADS))
        print('\tDebug = {}'.format(c.DEBUG))
        print('\tMulti test = {}'.format(c.MULTI_TEST))

        images_labels = load_images_lesion_labels(images_dir, lesion_dir, label_path)
        length = len(images_labels)
        true_labels = np.empty(shape=(length,), dtype=np.int32)
        ensemble = c.MULTI_TEST
        logits_ensemble = np.empty(shape=(length, ensemble), dtype=np.float32)

        _left_info = '{gpu ' + gpu_id + '['
        _right_info = ']' + self.mode_load_path + '}'
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('>', _left_info, _right_info), ' ',
                                               progressbar.SimpleProgress(), ' ',
                                               progressbar.Percentage(), ' ', progressbar.ETA()]).start()

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
                bar.update(i+1)
                # logits_ensemble[i] = np.random.random()
                # true_labels[i] = np.random.random()
                logits_ensemble[i] = logits[i-start]
                true_labels[i] = images_labels[i][2]
                # print('Image = {}, Ensemble = {},, True label = {}'.format(i,  logits_ensemble[i], true_labels[i]))

        # close bar
        bar.finish()
        # close session
        self.sess.close()

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
        return kappa, accuracy, mean_accuracy, class_accuracy, logits_ensemble

    def debug(self):
        data, labels = load_batch(c.BATCH_SIZE)
        self.model.debug(data, labels)


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')

    parser.add_argument('-g', '--gpu', type=str, nargs='*', choices=['0', '1', '2', '3'], help='the device id of gpu.')
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
    parser.add_argument('-', '--filters', type=str, default='',
                        help='the flag to control whether to use ImageNet pretrain model or not.')
    parser.add_argument('--multi_test', type=int, default=1, help='whether to use multi test or not')
    return parser.parse_args()


class ResultRecord(object):
    """
    Auxiliary class to record the validating or testing results.
    """

    def __init__(self, model='', pred_labels=None, kappa=0.0,
                 accuracy=0.0, mean_accuracy=0.0,
                 class_accuracy=None, dataset=None):
        self.model = model
        self.pred_labels = pred_labels
        self.kappa = kappa
        self.accuracy = accuracy
        self.mean_accuracy = mean_accuracy
        self.class_accuracy = class_accuracy
        self.dataset = dataset

    def __str__(self):
        _str = ('Model = {} \n' +
                '    kappa = {} \n' +
                '    accuracy = {} \n' +
                '    mean accuracy = {} \n' +
                '    class accuracy = {} \n' +
                '    dataset = {} \n').format(self.model, self.kappa, self.accuracy,
                                              self.mean_accuracy, self.class_accuracy, self.dataset)
        return _str

    def __le__(self, other):
        return self.kappa < other.kappa

    def __gt__(self, other):
        return self.kappa > other.kappa


class Producer(Process):
    """
    Producer Thread.
    """
    def __init__(self, thread_name, queue, model_folder, result_path, filter_models_txt):
        self.queue = queue
        self.is_run = True
        self.folder = model_folder
        self.result_path = result_path
        self.used_models = set()
        self.filter_models_txt = filter_models_txt

        db = shelve.open(self.result_path)
        for used_model in db:
            self.used_models.add(used_model)

        Process.__init__(self, name=thread_name)

    def terminate(self):
        self.is_run = False

    def run(self):
        print('{} is running.'.format(self.name))

        while True and self.is_run:
            print('scanning for {}'.format(self.folder))
            candidate_models = self._scanning_folder() - self.used_models

            for candidate in candidate_models:
                self.queue.put(candidate)
                print('Producer adds a new model: {}'.format(candidate))

            # update used_models
            self.used_models |= candidate_models

            # display results
            self._display()

            time.sleep(300)

    def _display(self):
        db = shelve.open(self.result_path)
        result_records = db.values()

        optimal = ResultRecord()
        for result_record in result_records:
            print('Producer displays: {}'.format(result_record))
            if optimal < result_record:
                optimal = result_record

        print('Producer, display optimal: {}'.format(optimal))

    def _filter_models(self):
        filter_models = set()
        if self.filter_models_txt:
            for line in open(self.filter_models_txt):
                print(line)
                line = line.rstrip()
                filter_models.add(line)
        return filter_models

    def _scanning_folder(self):
        filter_models = self._filter_models()
        current_model_sets = set()
        for _file in os.listdir(self.folder):
            if _file.startswith('model.ckpt-'):
                # ['model', 'ckpt-xxx', 'index' or 'meta' or 'data-00000-of-00001']
                splits = _file.split('.')
                checkpoint = splits[0] + '.' + splits[1]
                checkpoint_path = os.path.join(self.folder, checkpoint)
                if checkpoint_path not in filter_models:
                    current_model_sets.add(checkpoint_path)
        return current_model_sets


class Consumer(Process):
    def __init__(self, gpu, queue, result_path, num_epochs, architecture, metric_types, attentions):
        self.queue = queue
        self.result_path = result_path
        self.is_run = True
        self.num_epochs = num_epochs
        self.architecture = architecture
        self.metric_types = metric_types
        self.gpu_id = gpu
        self.attentions = attentions
        Process.__init__(self, name='Consumer_'+gpu)

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        print('{} is running, using gpu {}.'.format(self.name, self.gpu_id))

        while True and self.is_run:
            try:
                model_path = self.queue.get()
                driver = Driver(self.num_epochs, model_path, self.architecture, attention_names=self.attentions)
                kappa, accuracy, mean_accuracy, class_accuracy, pred_labels = driver.evaluate(
                    c.TEST_IMAGES_FOLDERS[c.SCALE_INDEX],
                    c.TEST_LABELS,
                    metric_types=self.metric_types,
                    batch_size=c.BATCH_SIZE
                )
                # kappa = np.random.random()
                # accuracy = np.random.random()
                # mean_accuracy = np.random.random()
                # class_accuracy = np.random.random()
                # pred_labels = []
                result_record = ResultRecord(
                    model=model_path,
                    pred_labels=pred_labels,
                    kappa=kappa,
                    accuracy=accuracy,
                    mean_accuracy=mean_accuracy,
                    class_accuracy=class_accuracy,
                    dataset=c.TEST_FOLDER
                )
                print('{}: {}'.format(self.name, result_record))

                del driver
                tf.reset_default_graph()

                db = shelve.open(self.result_path, writeback=True)
                db[model_path] = result_record
                db.close()

            except Exception('model error!') as e:
                print(e.message)

    def terminate(self):
        self.is_run = False


def main():
    ##
    # Handle command line input.
    ##

    load_path = None
    num_epochs = None
    architecture = None
    metric_types = [c.METRIC_ACCURACY]
    filter_txt = None
    gpus = '0'
    attentions = None

    # Args with Namespace
    args = parser_args()
    # Convert Namespace to dict by built-in function vars()
    for opt, arg in vars(args).items():
        if opt is 'gpu':
            gpus = arg
        if opt is 'train_dir':
            c.set_train_dir(arg)
        if opt is 'val_dir':
            c.set_validate_dir(arg)
        if opt is 'test_dir' and arg:
            c.set_test_dir(arg)
            c.IS_TRAINING = False
        if opt is 'architecture':
            architecture = arg
        if opt is 'attention':
            attentions = arg
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
        if opt is 'batch':
            assert arg > 0, 'batch size {} must > 0.'.format(arg)
            c.BATCH_SIZE = int(arg)
        if opt is 'pretrain':
            c.USE_PRETRAIN = arg
        if opt is 'multi_test':
            c.MULTI_TEST = arg
        if opt is 'filter' and arg:
            filter_txt = arg

    ##
    # Init and run the predictor
    ##
    c.set_save_dir('Scale_{}'.format('_'.join([str(x) for x in c.MULTI_SCALES])))
    assert load_path, 'In only testing setting, load_path {} must be specific and can not be None'.format(load_path)
    assert os.path.isdir(load_path), 'load_path {} must be a directory!'.format(load_path)

    result_path = os.path.join(c.RES_SAVE_DIR, os.path.split(c.TEST_FOLDER)[1])
    que = Queue()
    producer = Producer('Producer', que, load_path, result_path, filter_txt)

    consumers = []
    for gpu in gpus:
        consumer = Consumer(gpu, que, result_path, num_epochs, architecture, metric_types, attentions=attentions)
        consumers.append(consumer)

    # all processed start
    producer.start()
    for consumer in consumers:
        consumer.start()

    # all processes join
    producer.join()
    for consumer in consumers:
        consumer.join()


if __name__ == '__main__':
    main()
