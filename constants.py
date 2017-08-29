import os
from datetime import datetime
import shutil
import socket

##
# Data
##


def get_date_str():
    """
    :return: A string representing the current data/time that can be used as directory name.
    """
    return str(datetime.now()).replace(' ', '_').replace(':', '.')[:-10]


def get_dir(directory):
    """
    Creates the givens directory if it does not exits.

    :param directory: The path to the directory.
    :return: The path to the directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def clear_dir(directory):
    """
    Removes all files in the given directory.

    :param directory: The path to the directory
    :return: None
    """

    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)


def set_train_dir(directory):
    global TRAIN_FOLDER, TRAIN_IMAGES_FOLDERS, TRAIN_LABELS, MULTI_SCALES

    # check directory
    assert os.path.isdir(directory), '{} is not a directory!'.format(directory)

    # check whether there is a Label.csv file in this directory or not
    new_train_labels = os.path.join(directory, 'Label.csv')
    assert os.path.exists(new_train_labels), 'There is not label file {}!'.format(new_train_labels)

    # check whether there is scale folders in this directory or not
    new_scale_folders = []
    for scale in MULTI_SCALES:
        scale_folder = 'Image_' + str(scale) + 'x' + str(scale)
        new_scale_folder = os.path.join(directory, scale_folder)
        assert os.path.exists(new_scale_folder), '{} is not a directory!'.format(new_scale_folder)
        new_scale_folders.append(new_scale_folder)

    TRAIN_FOLDER = directory
    TRAIN_IMAGES_FOLDERS = new_scale_folders
    TRAIN_LABELS = new_train_labels


def set_validate_dir(directory):
    global VALIDATE_FOLDER, VALIDATE_IMAGES_FOLDERS, VALIDATE_LABELS, MULTI_SCALES

    # check directory
    assert os.path.isdir(directory), '{} is not a directory!'.format(directory)

    # check whether there is a Label.csv file in this directory or not
    new_validate_labels = os.path.join(directory, 'Label.csv')
    assert os.path.exists(new_validate_labels), 'There is not label file {}!'.format(new_validate_labels)

    # check whether there is scale folders in this directory or not
    new_scale_folders = []
    for scale in MULTI_SCALES:
        scale_folder = 'Image_' + str(scale) + 'x' + str(scale)
        new_scale_folder = os.path.join(directory, scale_folder)
        assert os.path.exists(new_scale_folder), '{} is not a directory!'.format(new_scale_folder)
        new_scale_folders.append(new_scale_folder)

    VALIDATE_FOLDER = directory
    VALIDATE_IMAGES_FOLDERS = new_scale_folders
    VALIDATE_LABELS = new_validate_labels


def set_test_dir(directory):
    """
    Edits all constants dependent on TEST_DIR.
    :param directory: The new test directory.
    :return: None
    """
    global TEST_FOLDER, TEST_IMAGES_FOLDERS, TEST_LABELS, MULTI_SCALES

    # check directory
    assert os.path.isdir(directory), '{} is not a directory!'.format(directory)

    # check whether there is a Label.csv file in this directory or not
    new_test_labels = os.path.join(directory, 'Label.csv')
    assert os.path.exists(new_test_labels), 'There is not label file {}!'.format(new_test_labels)

    # check whether there is scale folders in this directory or not
    new_scale_folders = []
    for scale in MULTI_SCALES:
        scale_folder = 'Image_' + str(scale) + 'x' + str(scale)
        new_scale_folder = os.path.join(directory, scale_folder)
        assert os.path.exists(new_scale_folder), '{} is not a directory!'.format(new_scale_folder)
        new_scale_folders.append(new_scale_folder)

    TEST_FOLDER = directory
    TEST_IMAGES_FOLDERS = new_scale_folders
    TEST_LABELS = new_test_labels


def set_save_dir(save_name):
    """
    Edits all constants dependent on TEST_DIR.
    :param save_name: The new save directory.
    :return: None
    """

    global SAVE_DIR, SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, RES_SAVE_DIR

    SAVE_NAME = save_name
    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
    RES_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Results/', SAVE_NAME))


ON_SERVER = socket.gethostname() == 'dl-T8520-G10'
if ON_SERVER:  # 119
    DATA_DIR = '/ssd/liuwen/kaggle/original'
    DATA_SPLIT_DIR = '/ssd/liuwen/kaggle/process_data'
else:
    DATA_DIR = '/home/imed269506/battleNet4/DATA/original'
    DATA_SPLIT_DIR = '/home/imed269506/battleNet4/DATA/process_data'

LABEL_CVS_PATH = os.path.join(DATA_DIR, 'label', 'train_label.csv')
DATA_IMAGES_FOLDER = os.path.join(DATA_DIR, 'train')
DATA_VAL_TEST_IMAGES_FOLDER = os.path.join(DATA_DIR, 'test', 'Images')
DATA_VAL_TEST_LABELS = os.path.join(DATA_DIR, 'test', 'label', 'test_label.csv')

NUM_CLASSES = 5


SHUFFLE = True
DEBUG = False

# Hyper-Parameters, Image_720x720
DECAY_STEP = 5000
DECAY_EPOCH = 2

LAMBDA = 1e-5
FINE_TUNE_LR = 0.0001  # 0 - 20,000
LEARNING_RATE = 0.00001  # 0 - 20,000
# FINE_TUNE_LR = 0.00001  # 0 - 20,000
# LEARNING_RATE = 0.0001  # 0 - 20,000

# Hyper-Parameters, Image_640x640
# FINE_TUNE_LR = 0.001  # 0 - 24,000
# LEARNING_RATE = 0.01  # 0 - 24,000
# FINE_TUNE_LR = 0.0001  # 150,000 - 300,000
# LEARNING_RATE = 0.001  # 150,000 - 300,000

MOMENTUM = 0.9

# MULTI SCALES
NUM_THREADS = 4
SCALE_INDEX = 0
MULTI_SCALES = [720]
MULTI_SCALES_FOLDERS = ['Image_' + str(crop) + 'x' + str(crop) for crop in MULTI_SCALES]

# Attention
# add attention module in ('Conv2d_2a_3x3', 'Conv2d_2b_3x3'), feature map scale 147 x 147 x 32
# add attention module in ('MaxPool_3a_3x3', 'Conv2d_4a_3x3'), feature map scale 73 x 73 x 80
# add attention module in ('MaxPool_5a_3x3', 'Mixed_5b'), feature map scale 35 x 35 x 320
# add attention module in ('Mixed_6a', 'PreAuxLogits'), feature map scale 17 x 17 x 1088
# add attention module in ('Mixed_7a', 'Conv2d_7b_1x1'), feature map scale 8 x 8 x 1536
ATTENTIONS = ['Conv2d_2a_3x3', 'MaxPool_3a_3x3', 'MaxPool_5a_3x3', 'Mixed_6a', 'Mixed_7a']

# Training
USE_PRETRAIN = 'imagenet'
BATCH_SIZE = 4
ATTENTION_BATCH_SIZE = 1
IS_TRAINING = True
TRAIN_FOLDER = get_dir(os.path.join(DATA_SPLIT_DIR, 'train'))
TRAIN_IMAGES_FOLDERS = [get_dir(os.path.join(TRAIN_FOLDER, scale_folder))
                        for scale_folder in MULTI_SCALES_FOLDERS]
TRAIN_LABELS = os.path.join(TRAIN_FOLDER, 'Label.csv')
TRAIN_LESION_IMAGES = 'data/images'
TRAIN_LESION_MAPS = 'data/lesions'

# Validate
VALIDATE_FOLDER = get_dir(os.path.join(DATA_SPLIT_DIR, 'validate'))
VALIDATE_IMAGES_FOLDERS = get_dir(os.path.join(VALIDATE_FOLDER, 'Images'))
VALIDATE_LABELS = os.path.join(VALIDATE_FOLDER, 'Label.csv')

# Testing
METRIC_ACCURACY = 'accuracy'
METRIC_KAPPA = 'kappa'
METRIC_SEN_SPE = 'sensitivity_specificity'
TEST_FOLDER = get_dir(os.path.join(DATA_SPLIT_DIR, 'test'))
TEST_IMAGES_FOLDERS = get_dir(os.path.join(TEST_FOLDER, 'Images'))
TEST_LABELS = os.path.join(TEST_FOLDER, 'Label.csv')

MULTI_TEST = 1

# root directory for all saved content
SAVE_DIR = get_dir('SAVE')
# inner directory to differentiate between runs
SAVE_NAME = 'Default/'
# directory for saved models
MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', SAVE_NAME))
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', SAVE_NAME))
# directory for saved images
RES_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Results/', SAVE_NAME))


STATS_FREQ = 10         # how often to print loss/train error stats, in # steps
SUMMARY_FREQ = 100      # how often to save the summaries, in # steps
VALIDATE_FREQ = 10000   # how often to test the model on test data, in # steps
MODEL_SAVE_FREQ = 1000   # how often to save the model, in # steps
