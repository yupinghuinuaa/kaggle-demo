import csv
import os
import shutil


folder_path = '/home/liuwen/ssd/kaggle/original'

csv_file_path= os.path.join(folder_path, 'label', 'test_val_label.csv')
test_validation_path = os.path.join(folder_path, 'test_val')

test_path = os.path.join(folder_path, 'test')
test_label_file_path = os.path.join(folder_path, 'label', 'test_label.csv')

validation_path = os.path.join(folder_path, 'validate')
validation_label_file_path = os.path.join(folder_path, 'label', 'validate_label.csv')


def load_label(label_csv_path):
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

def split_test_val():
    # [xxx.jpeg, 0, public]
    images_labels_usages = load_label(csv_file_path)
    validate_labels = []
    test_labels = []

    for image, label, usage in images_labels_usages:
        source_image_path = os.path.join(test_validation_path, image)

        if usage == 'Public':
            des_image_path = os.path.join(validation_path, image)
            validate_labels.append([image, label])
        elif usage == 'Private':
            des_image_path = os.path.join(test_path, image)
            test_labels.append([image, label])
        else:
            raise ValueError('usage {} must be Public or Private!'.format(usage))

        shutil.copy(source_image_path, des_image_path)
        print('copy {} ======> {}'.format(source_image_path, des_image_path))

    # write validate and test label to csv files.
    with open(validation_label_file_path, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(['image', 'level'])
        writer.writerows(validate_labels)
    print('writing validate label {} successfully!'.format(validation_label_file_path))

    with open(test_label_file_path, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerow(['image', 'level'])
        writer.writerows(test_labels)
    print('writing validate label {} successfully!'.format(test_label_file_path))


if __name__ == "__main__":
    split_test_val()