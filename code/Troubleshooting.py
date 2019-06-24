from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dropout
from keras.layers import Flatten, Dense
import os
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import read_images_labels as read

# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""         训练模型        """


# x_train, y_train = read_images_labels(data_dir='../train_images/', batch_size=10000)
x_test, y_test = read.read_images_labels(data_dir='E:/program/dataSet/Picture-classification/test_images/', batch_size=1200, shuffle=False)
# x_train_one = x_train * (1. / 255) - 0.5
x_test_one = x_test * (1. / 255) - 0.5


model = Sequential()
model.add(Conv2D(filters=48, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.load_weights("../flower10model.h5")
print("成功加载已有模型，开始检验准确率")


model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
# train_history = model.fit(x=x_train_one, y=y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)

# model.save_weights("flower10model.h5")
# print("保存刚训练的模型")
#
result = model.evaluate(x_test_one, y_test, verbose=1)
print('acc:', result[1])  # 模型准确率


"""         显示具体是哪一张图片预测错了      """


# x_ver_img = x_test
x_ver = x_test_one
y_ver = y_test

# print(y_ver)
# 获取预测的labels的值
prediction = model.predict_classes(x_ver)
prediction = prediction.tolist()
print("prediction:", prediction)

# 获取未经处理的labels
labels = []
for i in range(len(y_ver)):
    label = y_ver[i]
    for j in range(10):
        if label[j] == 1:
            labels.append(j)
print("labels:", labels)

# 获取到第n类图片是图片的总张数
label_0 = labels.count(0)
label_1 = label_0 + labels.count(1)
label_2 = label_1 + labels.count(2)
label_3 = label_2 + labels.count(3)
label_4 = label_3 + labels.count(4)
label_5 = label_4 + labels.count(5)
label_6 = label_5 + labels.count(6)
label_7 = label_6 + labels.count(7)
label_8 = label_7 + labels.count(8)
label_9 = label_8 + labels.count(9)
# print(label_7)

# 获取预测出错的图片索引
err_list = []
for j in range(len(prediction)):
    if prediction[j] != labels[j]:
        err_list.append(j)
print("err_list:", err_list)
print("images_number:", len(y_ver))
print("err_number:", len(err_list))
print("accuracy_rate:", 1 - len(err_list) / len(y_ver))  # 预测准确率

# 获取预测出错的图片在其类别中的索引
err_0 = []
err_1 = []
err_2 = []
err_3 = []
err_4 = []
err_5 = []
err_6 = []
err_7 = []
err_8 = []
err_9 = []
for err in err_list:
    if err < label_0:
        err_0.append(err + 1)
    if label_0 <= err < label_1:
        err_1.append(err - label_0 + 1)
    if label_1 <= err < label_2:
        err_2.append(err - label_1 + 1)
    if label_2 <= err < label_3:
        err_3.append(err - label_2 + 1)
    if label_3 <= err < label_4:
        err_4.append(err - label_3 + 1)
    if label_4 <= err < label_5:
        err_5.append(err - label_4 + 1)
    if label_5 <= err < label_6:
        err_6.append(err - label_5 + 1)
    if label_6 <= err < label_7:
        err_7.append(err - label_6 + 1)
    if label_7 <= err < label_8:
        err_8.append(err - label_7 + 1)
    if label_8 <= err:
        err_9.append(err - label_8 + 1)
print("err_0:", err_0)
print("err_1:", err_1)
print("err_2:", err_2)
print("err_3:", err_3)
print("err_4:", err_4)
print("err_5:", err_5)
print("err_6:", err_6)
print("err_7:", err_7)
print("err_8:", err_8)
print("err_9:", err_9)
print(len(err_5))

# type = []
# for label in labels:
#     type.append(labels_dict[label])
# print("种类", type)


# def plot_images_labels(images, labels):
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     num = len(images)
#     fig = plt.gcf()
#     fig.set_size_inches(12, 14)
#     if num % 5 == 0:
#         row = num // 5
#     else:
#         row = num // 5 + 1
#     for i in range(num):
#         ax = plt.subplot(row, 5, 1+i)
#         ax.imshow(images[i], cmap='binary')
#         title = 'label' + str(labels[i])
#         ax.set_title(title, fontsize=10)
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()

# plot_images_labels(x_ver_img, type)
