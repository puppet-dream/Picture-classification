import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_dir = '../train_images/'  # 图片所在文件夹地址
LABEL_FILE = '../labels.txt'   # 标签文件位置（自动生成）
IMAGE_SIZE = 32  # 图片默认大小
num_classes = 10  # 图片种类的数量
labels_dict = {}


def get_filepaths_and_labels(data_dir):
    """
    获取图片路径和labels
    :param data_dir:
    :return: [filepaths], [labels_dict: key标签,value索引]
    """
    if not os.path.exists(data_dir):
        raise ValueError('cannot find the dir: ' + data_dir)

    filepaths = []
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
    """
           对图片和labels进行预处理
           :param imgs
           :param labels
           :return [imgs], [relabels]
    """
    batch_size = len(imgs)
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


# 主函数
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


# 显示图片和labels
def plot_images_labels(images, labels):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    num = len(images)
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num % 5 == 0:
        row = num // 5
    else:
        row = num // 5 + 1
    for i in range(num):
        ax = plt.subplot(row, 5, 1+i)
        ax.imshow(images[i], cmap='binary')
        title = 'label:' + str(labels[i])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# 调试用
if __name__ == "__main__":
    # 读取图片和labels
    x_test, y_test = read_images_labels(data_dir=data_dir, batch_size=100, shuffle=False)
    x_test_one = x_test * (1. / 255) - 0.5  # 归一化处理

    # 获取未经处理的labels
    labels = []
    for i in range(len(y_test)):
        label = y_test[i]
        for j in range(10):
            if label[j] == 1:
                labels.append(j)
    print("labels:", labels)
    print(labels_dict)

    # 获取label对应的类型
    type=[]
    for label in labels:
        for key in labels_dict:
            if label == labels_dict[key]:
                type.append(key)
    plot_images_labels(x_test[:10], type)
