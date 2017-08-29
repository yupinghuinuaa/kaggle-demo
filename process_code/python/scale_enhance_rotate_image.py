import csv
import cv2
import glob
import os
import numpy as np
import concurrent.futures
import socket

scale = 500
pool_size = 20

def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

if socket.gethostname() == 'dl-T8520-G10':
    path = '/home/liuwen/ssd/kaggle/original/'
else:
    path = '/media/doubility/DATA/kaggle/original/'
train_crop_scale_enhance_rotate_path = get_dir(os.path.join(path, 'train_crop_scale_enhance_rotate'))
train_crop_scale_path = get_dir(os.path.join(path, 'train_crop_scale'))

test_crop_scale_enhance_rotate_path = get_dir(os.path.join(path, 'test_crop_scale_enhance_rotate'))
test_crop_scale_path = get_dir(os.path.join(path, 'test_crop_scale'))

validation_crop_scale_enhance_rotate_path = get_dir(os.path.join(path, 'validate_crop_scale_enhance_rotate'))
validation_crop_scale_path = get_dir(os.path.join(path, 'validate_crop_scale'))

train_label = os.path.join(path, 'label', 'train_label.csv')
train_aug_label = os.path.join(path, 'label', 'train_aug_label.csv')

validate_label = os.path.join(path, 'label', 'validate_label.csv')
validate_aug_label = os.path.join(path, 'label', 'validate_aug_label.csv')

test_label = os.path.join(path, 'label', 'test_label.csv')
test_aug_label = os.path.join(path, 'label', 'test_aug_label.csv')

assert os.path.exists(train_label), 'There is no train label {}'.format(train_label)
assert os.path.exists(validate_label), 'There is no train label {}'.format(validate_label)
assert os.path.exists(test_label), 'There is no train label {}'.format(test_label)


def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(1)   #
    r = (x>x.mean()/10).sum()/2
    if r == 0.0:
        s = 1.0
    else:
        s = scale*1.0/r
    return cv2.resize(img, (0,0), fx=s, fy=s)

def scale_enhance_rotate(img_path):
    try:
        a = cv2.imread(img_path)
        a = scaleRadius(a, scale)
        a_scale = a.copy()
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1]//2,a.shape[0]//2), int(scale*0.95),(1,1,1),-1,8,0)
        a = a*b + 128 * (1 - b)
        path_all = ''
        path_all_scale = ''
        if 'train' in img_path:
            path_all = os.path.join(train_crop_scale_enhance_rotate_path, img_path.split('/')[-1])
            path_all_scale = os.path.join(train_crop_scale_path, img_path.split('/')[-1])
        elif 'test' in img_path:
            path_all = os.path.join(test_crop_scale_enhance_rotate_path, img_path.split('/')[-1])
            path_all_scale = os.path.join(test_crop_scale_path, img_path.split('/')[-1])
        elif 'validate' in img_path:
            path_all = os.path.join(validation_crop_scale_enhance_rotate_path, img_path.split('/')[-1])
            path_all_scale = os.path.join(validation_crop_scale_path, img_path.split('/')[-1])
        else:
            raise ValueError('image path {} is error'.format(img_path))

        print(path_all)
        print(path_all_scale)
        cv2.imwrite(path_all, a)
        cv2.imwrite(path_all_scale, a_scale)

        # path_all: xxx.jpeg
        # image_prefix = str(path_all.split('.')[0])
        #
        # image_rotate_0 = image_prefix + '_0.jpeg'
        # image_rotate_90 = image_prefix + '_90.jpeg'
        # image_rotate_180 = image_prefix + '_180.jpeg'
        # image_rotate_270 = image_prefix + '_270.jpeg'
        #
        # cv2.imwrite(image_rotate_0, a)
        # cv2.imwrite(image_rotate_90, np.rot90(a, 1))
        # cv2.imwrite(image_rotate_180, np.rot90(a, 2))
        # cv2.imwrite(image_rotate_270, np.rot90(a, 3))
    except:
        print(img_path)


def generate_aug_labels(source_csv, des_csv):
    des_image_label_list = []
    with open(source_csv, encoding='utf-8') as csv_file:
        # read csv file
        reader = csv.reader(csv_file)
        next(reader)

        for row in reader:
            image = row[0]      # xxx.jpeg
            label = row[1]
            # image_name = image.split('.')[0]    # xxxx
            # image_rotate_0 = image_name + '_0.jpeg'
            # image_rotate_90 = image_name + '_90.jpeg'
            # image_rotate_180 = image_name + '_180.jpeg'
            # image_rotate_270 = image_name + '_270.jpeg'
            #
            # des_image_label_list.append([image_rotate_0, label])
            # des_image_label_list.append([image_rotate_90, label])
            # des_image_label_list.append([image_rotate_180, label])
            # des_image_label_list.append([image_rotate_270, label])
            des_image_label_list.append([image, label])

    with open(des_csv, 'w', newline='') as write_file:
        csv_writer = csv.writer(write_file)
        csv_writer.writerow(['image', 'level'])
        csv_writer.writerows(des_image_label_list)

    print('writing aug labels {}'.format(des_csv))
    return des_image_label_list


if __name__ == "__main__":

    images = glob.glob(path+"train_crop/*.jpeg") + glob.glob(path+'test_crop/*.jpeg') \
             + glob.glob(path+'validate_crop/*.jpeg')

    # images = glob.glob(path+'validate_crop/*.jpeg')
    # debug
    # scale_enhance_rotate(images[0])
    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        executor.map(scale_enhance_rotate, images)

    # generate aug labels.csv
    generate_aug_labels(train_label, train_aug_label)
    generate_aug_labels(validate_label, validate_aug_label)
    generate_aug_labels(test_label, test_aug_label)

