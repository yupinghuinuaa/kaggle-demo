import cv2
import numpy as np
import concurrent.futures as futures
import csv
import os
import shutil
import constants as c


def check_images_match_labels(images_labels, images_paths):
    """
    Check whether images match labels, for each image path in images_paths, there must be a pair(image, label)
    in images_labels.

    :param images_labels: the list of pairs(image, label).
    :type images_labels: list
    :param images_paths: the list contains all image paths.
    :type images_paths: list
    :return: True if matching, False otherwise.
    """

    assert len(images_labels) == len(images_paths), 'the lengths between image_labels ' \
                                                    '{} and images_paths {} do not ' \
                                                    'be equalled!'.format(len(images_labels), len(images_paths))

    flag = True
    for image_label, image_path in zip(images_labels, images_paths):
        image, label = image_label
        if image != image_path:
            print(image_label, image_path)
            flag = False
            break

    return flag


def load_label(label_csv_path=c.LABEL_CVS_PATH):
    image_label_list = []
    with open(label_csv_path, encoding='utf-8') as csv_file:
        # read csv file
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            row[0] = row[0] + '.jpeg' if not str.endswith(row[0], '.jpeg') else row[0]
            image_label_list.append(row)

    image_label_list.sort()
    return image_label_list


def split_data_by_label(data_folder=c.DATA_IMAGES_FOLDER,
                        label_csv_path=c.LABEL_CVS_PATH):
    assert os.path.exists(label_csv_path), 'label csv file {} does not exist!'.format(label_csv_path)
    assert os.path.isdir(data_folder), 'train folder {} is not a directory!'.format(data_folder)

    images_labels = load_label(label_csv_path)
    images_paths = sorted(os.listdir(data_folder))

    assert check_images_match_labels(images_labels, images_paths), 'images_labels dose not match to image_paths!'

    for idx, image_label in enumerate(images_labels):
        image, label = image_label
        original_image_path = os.path.join(data_folder, image)
        save_image_folder = os.path.join(c.TRAIN_IMAGES_FOLDERS[c.SCALE_INDEX], label)  # train/Images/01

        if not os.path.exists(save_image_folder):
            os.mkdir(save_image_folder)

        save_image_path = os.path.join(save_image_folder, image)
        # copy original image to destination image (train)
        shutil.copy(original_image_path, save_image_path)

        if idx % 100 == 0:
            print('copy {} ======> {}'.format(original_image_path, save_image_path))

    # copy original labels to
    shutil.copy(label_csv_path, c.TRAIN_LABELS)
    print('copy {} ======> {}'.format(label_csv_path, c.TRAIN_LABELS))


def split_val_test_by_label(validate_folder, test_folder):
    # [xxx.jpeg, 0, public]
    images_labels_usages = load_label(c.DATA_VAL_TEST_LABELS)
    validate_labels = []
    test_labels = []

    for image, label, usage in images_labels_usages:
        source_image_path = os.path.join(c.DATA_VAL_TEST_IMAGES_FOLDER, image)

        if usage == 'Public':
            des_image_path = os.path.join(validate_folder, image)
            validate_labels.append([image, label])
        elif usage == 'Private':
            des_image_path = os.path.join(test_folder, image)
            test_labels.append([image, label])
        else:
            raise ValueError('usage {} must be Public or Private!'.format(usage))

        shutil.copy(source_image_path, des_image_path)
        print('copy {} ======> {}'.format(source_image_path, des_image_path))

    # write validate and test label to csv files.
    with open(c.VALIDATE_LABELS, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(['image', 'level'])
        writer.writerows(validate_labels)
    print('writing validate label {} successfully!'.format(c.VALIDATE_LABELS))

    with open(c.TEST_LABELS, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(['image', 'level'])
        writer.writerows(test_labels)
    print('writing validate label {} successfully!'.format(c.TEST_LABELS))


def _process_resize(source_des_paths_tuples, scale):
    length = len(source_des_paths_tuples)
    for i, (source, des) in enumerate(source_des_paths_tuples):
        # print(c.CROP_HEIGHT, c.CROP_WIDTH)
        source_image = cv2.imread(source)
        resize_image = cv2.resize(source_image, scale)
        cv2.imwrite(des, resize_image)

        if (i + 1) % 100 == 0 or i == length - 1:
            print('    \tprocess = {}, resizing {} / {} images, in scale ({}, {})'.format(os.getpid(),
                                                                                          i + 1, length, scale[0],
                                                                                          scale[1]))


def multi_processes_resize(source_folder, des_folder, scale):
    from multiprocessing import cpu_count
    num_process = min(20, cpu_count())

    if not os.path.exists(des_folder):
        os.mkdir(des_folder)

    source_des_paths_tuples = []
    scale_lists = []
    for image in os.listdir(source_folder):
        source_image = os.path.join(source_folder, image)
        des_image = os.path.join(des_folder, image)
        source_des_paths_tuples.append((source_image, des_image))
        scale_lists.append(scale)

    sub_source_des_paths_tuples = np.array_split(source_des_paths_tuples, num_process)

    total = len(sub_source_des_paths_tuples)
    with futures.ProcessPoolExecutor(max_workers=num_process) as pool:
        pool.map(_process_resize, sub_source_des_paths_tuples, scale_lists)
    print('    resizing total = {} images!'.format(total))


if __name__ == '__main__':
    # image_label_list = load_label('trainLabels.csv')
    #
    # for row in image_label_list:
    #     print(row)
    #     break
    # print('training data = {}'.format(len(image_label_list)))
    # split_val_test_by_label()
    # split_data_by_label()

    ## split_data_by_label()
    # split_data_by_label()

    ###### image resize
    ## Original

    if c.ON_SERVER:
        original_folder = '/home/liuwen/ssd/kaggle/original'
        des_folder = '/home/liuwen/ssd/kaggle/process_data'
    else:
        original_folder = '/media/doubility/NUMEROUS/kaggle/original'
        des_folder = '/media/doubility/NUMEROUS/kaggle/process_data'

    train_crop_scale_enhance_rotate_path = os.path.join(original_folder, 'train_crop_scale_enhance_rotate')
    train_aug_label = os.path.join(original_folder, 'label', 'train_aug_label.csv')

    validation_crop_scale_enhance_rotate_path = os.path.join(original_folder, 'validate_crop_scale_enhance_rotate')
    validate_aug_label = os.path.join(original_folder, 'label', 'validate_aug_label.csv')

    test_crop_scale_enhance_rotate_path = os.path.join(original_folder, 'test_crop_scale_enhance_rotate')
    test_aug_label = os.path.join(original_folder, 'label', 'test_aug_label.csv')

    ## Destination
    train_des_image_folder = os.path.join(des_folder, 'train')  # xxx/train/
    train_des_image_label = os.path.join(des_folder, 'train', 'Label.csv')  # xxx/train/Label.csv

    validate_des_image_folder = os.path.join(des_folder, 'validate')  # xxx/validate/
    validate_des_image_label = os.path.join(des_folder, 'validate', 'Label.csv')  # xxx/validate/Label.csv

    test_des_image_folder = os.path.join(des_folder, 'test')  # xxx/test/
    test_des_image_label = os.path.join(des_folder, 'test', 'Label.csv')  # xxx/test/Label.csv

    originals = [(train_crop_scale_enhance_rotate_path, train_aug_label),
                 (validation_crop_scale_enhance_rotate_path, validate_aug_label),
                 (test_crop_scale_enhance_rotate_path, test_aug_label)]

    des = [(train_des_image_folder, train_des_image_label),
           (validate_des_image_folder, validate_des_image_label),
           (test_des_image_folder, test_des_image_label)]
    #
    # originals = [(validation_crop_scale_enhance_rotate_path, validate_aug_label)]
    # des = [(validate_des_image_folder, validate_des_image_label)]

    for original, des in zip(originals, des):
        source_folder, source_label = original
        des_folder, des_label = des

        for scale in [(800, 800)]:
            scale_folder = 'Image_' + str(scale[0]) + 'x' + str(scale[1])

            des_scale_folder = os.path.join(des_folder, scale_folder)  # xxx/train/Image_224x224
            if not os.path.exists(des_scale_folder):
                os.makedirs(des_scale_folder)

            print('{} is resizing to {}'.format(source_folder, des_scale_folder))

            multi_processes_resize(source_folder, des_scale_folder, scale)

        shutil.copy(source_label, des_label)
        print('copy label from {} =====> {}'.format(source_label, des_label))

        # source_folder = '/media/doubility/DATA/kaggle/original/train'
        # des_folder = '/media/doubility/DATA/kaggle/original/resize_temp'
        #
        # multi_processes_resize(source_folder, des_folder)
