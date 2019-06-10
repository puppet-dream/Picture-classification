import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np


data_dir = '../train_images/'
LABEL_FILE = '../labels.txt'
# 图片默认大小
IMAGE_SIZE = 32
num_classes = 10


def get_filepaths_and_labels(data_dir):
    """
    获取图片路径和labels
    :param data_dir:
    :return: [filepaths], [labels_dict: key标签,value索引]
    """
    if not os.path.exists(data_dir):
        raise ValueError('cannot find the dir: ' + data_dir)

    filepaths = []
    labels_dict = {}

    index = 0
    for labeldir in os.listdir(data_dir):
        namedir = os.path.join(data_dir, labeldir)
        if os.path.isfile(namedir):
            continue
        for file in os.listdir(namedir):
            file = os.path.join(namedir, file)

            # 小于4k 的图片可能不完整不要
            # if os.path.getsize(file) / 1024 < 4:
            #     continue
            filepaths.append(file)
            if labeldir not in labels_dict:
                labels_dict[labeldir] = index
                index = index + 1
    print(labels_dict)
    return filepaths, labels_dict


def write_label_file(labels_dict, label_file):
    """
    将label和其索引存到文件
    :param labels_dict:
    :param label_file:
    :return:
    """
    with tf.gfile.Open(label_file, 'w') as f:
        for label in labels_dict:
            num = labels_dict[label]
            f.write('%d:%s\n' % (num, label))


def get_images_labels(filepaths, labels_dict, batch_size):
    """
       获取图片和label
       :param filepaths
       :param labels_dict
       :param batch_size
       :return [imgs], [labels]
    """
    imgs = []
    labels = []
    batch_size = min(len(filepaths), batch_size)
    print("图片数量：", len(filepaths))
    for j in range(batch_size):
        img = Image.open(filepaths[j])
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)

        # 获取目录名作为labels
        img_label = os.path.split(os.path.dirname(filepaths[j]))[1]
        img_label = labels_dict[img_label]

        imgs.append(img)
        labels.append(img_label)
    imgs = np.array(imgs)
    return imgs, labels


def re_imgs_labes(imgs, labels):
    batch_size = len(imgs)

    # 图片一致化
    imgs = imgs.reshape([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    # reimgs = imgs * (1. / 255) - 0.5

    # 将labels转为ont-hot编码
    labels = tf.one_hot(labels, num_classes, 1, 0)
    labels = tf.cast(labels, dtype=tf.int32)
    labels = tf.reshape(labels, [batch_size, num_classes])

    # 将labels转numpy数组类型
    sess = tf.Session()
    with sess.as_default():
       relabels = labels.eval()

    return imgs, relabels


def read_images_labels(data_dir, batch_size=1000, shuffle=True):
    # 获取路径和label字典
    data_paths, labels_dict = get_filepaths_and_labels(data_dir)

    # 根据图片来源判断是否打乱
    if shuffle:
        random.seed(0)
        random.shuffle(data_paths)
    filepath = data_paths
    imgs, labels = get_images_labels(filepath, labels_dict, batch_size)
    # print(imgs[0])
    # print(labels[0])

    reimgs, relabels = re_imgs_labes(imgs, labels)

    # 将label字典写入文件
    write_label_file(labels_dict, LABEL_FILE)
    print('finsh')
    return reimgs, relabels


if __name__ == "__main__":
    read_images_labels(data_dir)